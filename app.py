import os
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import datetime
from environment.cpu_env import CPUEnvironment
from environment.workload_generator import WorkloadGenerator
from scheduler.rl_scheduler import DQNScheduler
from scheduler.traditional_scheduler import FCFSScheduler, SJFScheduler, RoundRobinScheduler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.jinja_env.globals.update(now=datetime.datetime.now)
db_url = os.environ.get("DATABASE_URL", 'sqlite:///database.db')
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

if db_url:
    logger.info(f"Database URL configured successfully")
else:
    logger.warning("DATABASE_URL not found in environment variables")

from database import db, User, SimulationConfig, SimulationResult
db.init_app(app)

with app.app_context():
    db.create_all()

env = None
rl_scheduler = None
traditional_schedulers = {}
simulation_results = {}
current_workload = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulation', methods=['GET', 'POST'])
def simulation():
    loaded_config = session.pop('loaded_config', None)
    default_values = {}
    
    if loaded_config:
        default_values = loaded_config
        flash('Loaded configuration applied. Submit the form to start the simulation.', 'info')
    
    if request.method == 'POST':
        try:
            num_cpus = int(request.form.get('num_cpus', 4))
            num_tasks = int(request.form.get('num_tasks', 100))
            workload_type = request.form.get('workload_type', 'trading')
            global env, current_workload
            workload_generator = WorkloadGenerator(workload_type=workload_type)
            current_workload = workload_generator.generate_workload(num_tasks)
            
            env = CPUEnvironment(
                num_cpus=num_cpus,
                workload=current_workload,
                reward_latency_weight=float(request.form.get('reward_latency_weight', 0.7)),
                reward_throughput_weight=float(request.form.get('reward_throughput_weight', 0.3))
            )
            
            global rl_scheduler
            rl_scheduler = DQNScheduler(
                state_size=env.observation_space_size,
                action_size=env.action_space_size,
                learning_rate=float(request.form.get('learning_rate', 0.001)),
                gamma=float(request.form.get('gamma', 0.99)),
                epsilon=float(request.form.get('epsilon', 1.0)),
                epsilon_min=float(request.form.get('epsilon_min', 0.01)),
                epsilon_decay=float(request.form.get('epsilon_decay', 0.995)),
                memory_size=int(request.form.get('memory_size', 10000)),
                batch_size=int(request.form.get('batch_size', 64))
            )
            
            global traditional_schedulers
            traditional_schedulers = {
                'FCFS': FCFSScheduler(),
                'SJF': SJFScheduler(),
                'RoundRobin': RoundRobinScheduler(time_quantum=int(request.form.get('time_quantum', 2)))
            }
            
            logger.info("Simulation environment and schedulers initialized successfully")
           
            return redirect(url_for('train'))
        except Exception as e:
            logger.error(f"Error initializing simulation: {str(e)}")
            flash(f'Error initializing simulation: {str(e)}', 'danger')
            return render_template('simulation.html', default_values=default_values)
    
    return render_template('simulation.html', default_values=default_values)

import time

@app.route('/train', methods=['GET', 'POST'])
def train():
    global env, rl_scheduler, simulation_results

    if env is None or rl_scheduler is None:
        flash('Please configure simulation parameters first.', 'warning')
        return redirect(url_for('simulation'))
    
    if request.method == 'POST':
        try:
            requested_episodes = int(request.form.get('num_episodes', 100))
            num_episodes = min(requested_episodes, 5)
            
            if requested_episodes > 5:
                flash(f"For web demonstration, limited training to 5 episodes (you requested {requested_episodes}).", "info")
            logger.info(f"Starting RL training for {num_episodes} episodes...")
            
            if hasattr(rl_scheduler, 'batch_size'):
                rl_scheduler.batch_size = min(rl_scheduler.batch_size, 32)
            max_training_time = 25  # seconds
            max_steps_per_episode = 200
            
            start_time = time.time()
            training_history = rl_scheduler.train(
                env, 
                num_episodes, 
                max_time=max_training_time,
                max_steps_per_episode=max_steps_per_episode
            )
            elapsed_time = time.time() - start_time
          
            completed_episodes = len(training_history['rewards'])
            logger.info(f"RL training completed {completed_episodes}/{num_episodes} episodes in {elapsed_time:.2f} seconds")
            
            if completed_episodes < num_episodes:
                flash(f"Training time limit reached. Completed {completed_episodes}/{num_episodes} episodes.", "info")
            
            logger.info("Running simulations with all schedulers...")
            simulation_results = run_simulations()
            logger.info("Simulations completed")
            
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['rewards'])
            plt.title('Training Rewards Over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            training_plot = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return render_template('results.html', 
                                results=simulation_results, 
                                training_plot=training_plot)
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            flash(f'Error during training: {str(e)}', 'danger')
            return redirect(url_for('simulation'))
    
    return render_template('simulation.html', train=True)

@app.route('/results')
def results():
    global simulation_results
    
    if not simulation_results:
        flash('No simulation results available. Please run a simulation first.', 'warning')
        return redirect(url_for('simulation'))
    
    return render_template('results.html', results=simulation_results)

def run_simulations():
    global env, rl_scheduler, traditional_schedulers, current_workload
    
    results = {}
    
    env.reset()
    rl_state = env.get_state()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action = rl_scheduler.act(rl_state, training=False)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        rl_state = next_state
        step_count += 1
    
    results['RL'] = {
        'avg_latency': float(env.get_average_latency()),
        'throughput': float(env.get_throughput()),
        'cpu_utilization': float(env.get_cpu_utilization()),
        'total_reward': float(total_reward),
        'completed_tasks': int(env.get_completed_tasks())
    }
    
    for name, scheduler in traditional_schedulers.items():
        env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = scheduler.select_action(env)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        results[name] = {
            'avg_latency': float(env.get_average_latency()),
            'throughput': float(env.get_throughput()),
            'cpu_utilization': float(env.get_cpu_utilization()),
            'total_reward': float(total_reward),
            'completed_tasks': int(env.get_completed_tasks())
        }
    
    return results

@app.route('/api/compare_metrics')
def compare_metrics():
    global simulation_results
    
    if not simulation_results:
        return jsonify({'error': 'No simulation results available'})
    
    metrics = {}
   
    latency_data = {scheduler: float(results['avg_latency']) 
                    for scheduler, results in simulation_results.items()}

    throughput_data = {scheduler: float(results['throughput']) 
                      for scheduler, results in simulation_results.items()}
    
    utilization_data = {scheduler: float(results['cpu_utilization']) 
                       for scheduler, results in simulation_results.items()}
    
    return jsonify({
        'latency': latency_data,
        'throughput': throughput_data,
        'utilization': utilization_data
    })

@app.route('/api/configs', methods=['GET'])
def api_configs():
    try:
        default_user = get_or_create_default_user()
        configs = SimulationConfig.query.filter_by(user_id=default_user.id).all()
        configs_data = [{
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'created_at': config.created_at.isoformat(),
            'params': {
                'num_cpus': config.num_cpus,
                'num_tasks': config.num_tasks,
                'workload_type': config.workload_type,
                'reward_weights': {
                    'latency': float(config.reward_latency_weight),
                    'throughput': float(config.reward_throughput_weight)
                },
                'rl_params': {
                    'learning_rate': float(config.learning_rate),
                    'gamma': float(config.gamma),
                    'epsilon': float(config.epsilon),
                    'epsilon_min': float(config.epsilon_min),
                    'epsilon_decay': float(config.epsilon_decay),
                    'memory_size': config.memory_size,
                    'batch_size': config.batch_size
                }
            }
        } for config in configs]
        
        return jsonify({
            'status': 'success',
            'configs': configs_data
        })
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/results/<int:config_id>', methods=['GET'])
def api_results(config_id):
    try:
        results = SimulationResult.query.filter_by(config_id=config_id).all()
        
        if not results:
            return jsonify({
                'status': 'error',
                'message': f'No results found for configuration ID {config_id}'
            }), 404
        
        results_data = [{
            'id': result.id,
            'scheduler_type': result.scheduler_type,
            'run_date': result.run_date.isoformat(),
            'metrics': {
                'avg_latency': float(result.avg_latency),
                'throughput': float(result.throughput),
                'cpu_utilization': float(result.cpu_utilization),
                'total_reward': float(result.total_reward),
                'completed_tasks': result.completed_tasks
            }
        } for result in results]
        
        return jsonify({
            'status': 'success',
            'config_id': config_id,
            'results': results_data
        })
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/db_management')
def db_management():
    with app.app_context():
        default_user = get_or_create_default_user()
        configs = SimulationConfig.query.filter_by(user_id=default_user.id).all()
        results = SimulationResult.query.options(
            db.joinedload(SimulationResult.config)
        ).join(SimulationConfig).filter(SimulationConfig.user_id == default_user.id).all()
    
    return render_template('db_management.html', configs=configs, results=results)

@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        config_data = {
            'name': request.form.get('name'),
            'description': request.form.get('description'),
            'num_cpus': int(request.form.get('num_cpus', 4)),
            'num_tasks': int(request.form.get('num_tasks', 100)),
            'workload_type': request.form.get('workload_type', 'trading'),
            'reward_latency_weight': float(request.form.get('reward_latency_weight', 0.7)),
            'reward_throughput_weight': float(request.form.get('reward_throughput_weight', 0.3)),
            'max_steps': int(request.form.get('max_steps', 1000)),
            'learning_rate': float(request.form.get('learning_rate', 0.001)),
            'gamma': float(request.form.get('gamma', 0.99)),
            'epsilon': float(request.form.get('epsilon', 1.0)),
            'epsilon_min': float(request.form.get('epsilon_min', 0.01)),
            'epsilon_decay': float(request.form.get('epsilon_decay', 0.995)),
            'memory_size': int(request.form.get('memory_size', 10000)),
            'batch_size': int(request.form.get('batch_size', 64)),
            'time_quantum': int(request.form.get('time_quantum', 2))
        }
        
        default_user = get_or_create_default_user()
       
        new_config = SimulationConfig(
            user_id=default_user.id,
            created_at=datetime.datetime.utcnow(),
            **config_data
        )
      
        db.session.add(new_config)
        db.session.commit()
        
        flash('Configuration saved successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving configuration: {str(e)}")
        flash(f'Error saving configuration: {str(e)}', 'danger')
    
    return redirect(url_for('db_management'))

@app.route('/load_config/<int:config_id>')
def load_config(config_id):
    try:
        config = SimulationConfig.query.get_or_404(config_id)
        config_data = {
            'num_cpus': config.num_cpus,
            'num_tasks': config.num_tasks,
            'workload_type': config.workload_type,
            'reward_latency_weight': config.reward_latency_weight,
            'reward_throughput_weight': config.reward_throughput_weight,
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'epsilon': config.epsilon,
            'epsilon_min': config.epsilon_min,
            'epsilon_decay': config.epsilon_decay,
            'memory_size': config.memory_size,
            'batch_size': config.batch_size,
            'time_quantum': config.time_quantum
        }
      
        session['loaded_config'] = config_data
        
        flash(f'Configuration "{config.name}" loaded successfully!', 'success')
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        flash(f'Error loading configuration: {str(e)}', 'danger')
    
    return redirect(url_for('simulation'))

@app.route('/view_config/<int:config_id>')
def view_config(config_id):
    config = SimulationConfig.query.get_or_404(config_id)
    return render_template('view_config.html', config=config)

@app.route('/delete_config/<int:config_id>')
def delete_config(config_id):
    try:
        config = SimulationConfig.query.get_or_404(config_id)
        db.session.delete(config) 
        db.session.commit()
        flash('Configuration deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting configuration: {str(e)}")
        flash(f'Error deleting configuration: {str(e)}', 'danger')
    
    return redirect(url_for('db_management'))

@app.route('/view_result/<int:result_id>')
def view_result(result_id):
    result = SimulationResult.query.options(
        db.joinedload(SimulationResult.config)
    ).get_or_404(result_id)
  
    plots = {}
    if result.training_history and 'rewards' in result.training_history:
        plt.figure(figsize=(10, 6))
        plt.plot(result.training_history['rewards'])
        plt.title('Training Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plots['training'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    return render_template('view_result.html', result=result, plots=plots)

@app.route('/delete_result/<int:result_id>')
def delete_result(result_id):
    try:
        result = SimulationResult.query.get_or_404(result_id)
        db.session.delete(result)
        db.session.commit()
        flash('Result deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting result: {str(e)}")
        flash(f'Error deleting result: {str(e)}', 'danger')
    
    return redirect(url_for('db_management'))

@app.route('/save_current_results', methods=['POST'])
def save_current_results():
    global simulation_results, env, current_workload
    
    if not simulation_results:
        flash('No simulation results available to save!', 'warning')
        return redirect(url_for('results'))
    
    try:
        default_user = get_or_create_default_user()
        config_name = request.form.get('config_name', f'Simulation {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
        num_cpus = 4
        num_tasks = 100
        reward_latency_weight = 0.7
        reward_throughput_weight = 0.3
        if env is not None:
            num_cpus = env.num_cpus
            reward_latency_weight = env.reward_latency_weight
            reward_throughput_weight = env.reward_throughput_weight
            
        if current_workload is not None:
            num_tasks = len(current_workload)
        config = SimulationConfig(
            name=config_name,
            description=request.form.get('description', 'Auto-saved from simulation'),
            user_id=default_user.id,
            num_cpus=num_cpus,
            num_tasks=num_tasks,
            workload_type=request.form.get('workload_type', 'trading'),
            reward_latency_weight=reward_latency_weight,
            reward_throughput_weight=reward_throughput_weight,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            time_quantum=2
        )
        
        db.session.add(config)
        db.session.flush() 
        for scheduler_type, result_data in simulation_results.items():
            result = SimulationResult(
                config_id=config.id,
                scheduler_type=scheduler_type,
                avg_latency=float(result_data['avg_latency']),
                throughput=float(result_data['throughput']),
                cpu_utilization=float(result_data['cpu_utilization']),
                total_reward=float(result_data['total_reward']),
                completed_tasks=int(result_data['completed_tasks']),
                training_history=rl_scheduler.get_training_history() if scheduler_type == 'RL' and rl_scheduler else {'rewards': []}
            )
            db.session.add(result)
        
        db.session.commit()
        flash('Results saved successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving results: {str(e)}")
        flash(f'Error saving results: {str(e)}', 'danger')
    
    return redirect(url_for('results'))

def get_or_create_default_user():
    default_user = User.query.filter_by(username='default').first()
    if not default_user:
        default_user = User(
            username='default',
            email='default@example.com',
            password_hash='not_set' 
        )
        db.session.add(default_user)
        db.session.commit()
    return default_user

def load_workload(dataset="synthetic", num_processes=5000, google_csv="data/google/google_workload.csv"):
    if dataset == "synthetic":
        return generate_synthetic_processes(num_processes) # YOUR EXISTING CODE
        
    elif dataset == "google":
        df = pd.read_csv(google_csv)
        processes = []
        for _, row in df.iterrows():
            # NOTE: Adapt `Process` to whatever class/dict your codebase uses
            processes.append(Process(
                pid=int(row['pid']),
                arrival_time=int(row['arrival_time']),
                burst_time=int(row['burst_time']),
                priority=int(row['priority'])
            ))
        return processes
    else:
        raise ValueError("Invalid dataset type")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
