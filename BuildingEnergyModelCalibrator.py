from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from eppy.modeleditor import IDF
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime
import os
import re
import json

class BuildingEnergyModelCalibrator:
    def __init__(self, model_url="http://localhost:11434/", model_name="llama3.2:3b"):
        """Initialize the model and optimization components."""
        self.model = ChatOllama(model=model_name, base_url=model_url)
        self.chat_history = []
        self.optimizer_llm_dict = {
            "model_type": "llama3.2",
            "temperature": 1.0,
            "max_tokens": 1024,
            "top_p": 1,
            "stop": None,
        }
        
    def run_building_simulation(self, schedule):
        """Run building simulation with given schedule."""
        # Update the IDF file with the new schedule
        idf_path = "./ep-model/updated_ep_opt.idf"
        schedule_names =  ["BLDG_LIGHT_SCH", "BLDG_OCC_SCH", "BLDG_EQUIP_SCH"]
        self.revise_schedule_compact(idf_path, schedule_names, schedule)
        
        # Run simulation and get results
        updated_idf_path = "./ep-model/updated_ep.idf"
        return self.run_simulation(updated_idf_path)
        
    # def evaluate_loss(self, simulated_energy_consumption, ground_truth):
    #     """Calculate loss between simulation result and ground truth."""
        
    #     return np.mean(np.abs(np.array(simulated_energy_consumption) - np.array(ground_truth)))
    

    def evaluate_loss(self, simulated_energy_consumption, ground_truth):
        """Calculate CVRMSE between simulation result and ground truth."""
        
        # Convert inputs to numpy arrays
        sim = np.array(simulated_energy_consumption)
        gt = np.array(ground_truth)
        
        # Calculate the difference between simulated and ground truth values
        diff = sim - gt
        
        # Calculate the mean of the ground truth values
        mean_gt = np.mean(gt)
        
        # Calculate the Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean(diff**2))
        
        # Calculate the Coefficient of Variation of RMSE (CVRMSE)
        cvrmse = (rmse / mean_gt) * 100  # Multiply by 100 to express as a percentage
        
        return cvrmse

    def optimize_parameters(self, ground_truth, num_iterations=50, num_starting_points=5):
        """Optimize building parameters to match ground truth."""
        # Initialize results storage
        datetime_str = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.save_folder = f"optimization_results_{datetime_str}"
        os.makedirs(self.save_folder, exist_ok=True)
        
        # Generate starting points
        np.random.seed(42)
        old_value_pairs_set = set()
        
        # Generate initial schedules (24 values between 0 and 1)
        for start_point in range(num_starting_points):
            schedule = np.random.uniform(0, 1, 24)
            energy_consumption = self.run_building_simulation(schedule)
            loss = self.evaluate_loss(energy_consumption, ground_truth)
            old_value_pairs_set.add(tuple(list(schedule) + [loss]))
            
            # Log initial schedule for each starting point
            self._log_iteration(f"start_point_{start_point}", schedule, energy_consumption, loss)
        
        best_loss = float('inf')
        best_schedule = None
        
        # Optimization loop
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Generate meta prompt
            meta_prompt = self._generate_meta_prompt(old_value_pairs_set, ground_truth)
            
            # Get new proposals from LLM
            new_schedules = self._get_llm_proposals(meta_prompt)
            print(new_schedules)
            # Evaluate new proposals
            for schedule in new_schedules:
                energy_consumption = self.run_building_simulation(schedule)
                loss = self.evaluate_loss(energy_consumption, ground_truth)
                old_value_pairs_set.add(tuple(list(schedule) + [loss]))
                
                # Log each proposal
                self._log_iteration(f"iteration_{iteration+1}", schedule, energy_consumption, loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_schedule = schedule
                    print(f"New best loss: {best_loss}")
            
            # Save intermediate results
            self._save_results(self.save_folder, {
                'iteration': iteration,
                'best_loss': best_loss,
                'best_schedule': best_schedule.tolist() if best_schedule is not None else None,
                'current_pairs': list(old_value_pairs_set)
            })
        
        return best_schedule, best_loss

    def _generate_meta_prompt(self, old_value_pairs_set, ground_truth):
        """Generate meta prompt for LLM."""
        pairs_str = "\n".join([
            f"schedule: {list(pair[:-1])}\nloss: {pair[-1]}"
            for pair in sorted(old_value_pairs_set, key=lambda x: x[-1])[:10]
        ])
        
        return f"""
        Help me optimize a 24-hour building occupancy schedule to match a target energy consumption pattern.
        The schedule should contain 24 values between 0 and 1 representing hourly occupancy rates.
        
        Previous attempts and their losses:
        {pairs_str}
        
        Target energy consumption pattern:
        {ground_truth}
        
        Propose a new schedule that might achieve lower loss. Output should be exactly 24 numbers between 0 and 1.
        Consider typical building occupancy patterns (e.g., higher during work hours, lower at night).
        """

    def _get_llm_proposals(self, meta_prompt, num_proposals=5):
        """Get new schedule proposals from LLM."""
        proposals = []
        for _ in range(num_proposals):
            response = self.model.invoke(meta_prompt)
            try:
                # Extract numbers from response and validate
                numbers = [float(x) for x in re.findall(r"0?\.[0-9]+|[01]", response.content)]
                if len(numbers) >= 24:
                    schedule = np.clip(numbers[:24], 0, 1)
                    proposals.append(schedule)
            except:
                continue
        return proposals

    def _save_results(self, save_folder, results):
        """Save optimization results to file."""
        with open(f"{save_folder}/results.json", 'w') as f:
            json.dump(results, f, indent=4)

    def plot_comparison(self, best_schedule, ground_truth):
        """Plot comparison between best schedule and ground truth."""
        plt.figure(figsize=(12, 6))
        plt.plot(range(24), ground_truth, 'b-', label='Ground Truth', marker='o')
        plt.plot(range(24), self.run_building_simulation(best_schedule), 'r--', 
                label='Best Simulation', marker='x')
        plt.xlabel('Hour')
        plt.ylabel('Energy Consumption (kWh)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def revise_schedule_compact(self, idf_path: str, schedule_names: list, hourly_values: list):
        """
        Revises multiple Schedule:Compact objects in the IDF file with the provided hourly values.

        Parameters:
        - idf_path: Path to the IDF file.
        - schedule_names: List of Schedule:Compact object names to modify.
        - hourly_values: List of 24 float values representing hourly schedule.
        """
        # Validate input
        if len(hourly_values) != 24:
            raise ValueError("Hourly values list must contain exactly 24 values.")
        if not isinstance(schedule_names, list) or not schedule_names:
            raise ValueError("schedule_names must be a non-empty list.")

        # Load the IDF file
        IDF.setiddname("./ep-model/Energy+.idd")  
        self.idf = IDF(idf_path)

        # Find and update the Schedule:Compact objects
        schedules = self.idf.idfobjects['SCHEDULE:COMPACT']
        for schedule_name in schedule_names:
            for schedule in schedules:
                if schedule.Name == schedule_name:
                    # Generate the "Until" time fields
                    until_times = [f"Until: {i + 1:02}:00" for i in range(24)]

                    # Update the fields in Schedule:Compact
                    for i in range(24):
                        until_field = f"Field_{3 + (i * 2)}"  # Alternating "Until" fields
                        value_field = f"Field_{4 + (i * 2)}"  # Corresponding value fields

                        # Format values properly
                        formatted_value = f"{hourly_values[i]:.2f}".lstrip('0') if hourly_values[i] != 0 else "0.0"

                        setattr(schedule, until_field, until_times[i])  # Set "Until" time
                        setattr(schedule, value_field, formatted_value)  # Set value

                    print(f"Updated Schedule:Compact {schedule_name} with new hourly values.")
                    break
            else:
                print(f"Schedule:Compact with name '{schedule_name}' not found.")
                continue

        # Save the updated IDF
        updated_idf_path = "./ep-model/updated_ep.idf"
        self.idf.saveas(updated_idf_path)
        print(f"Updated IDF file saved at {updated_idf_path}")

    def run_simulation(self, idf_path: str):
        """Run the simulation using the updated IDF file and process the generated energy data."""
        # Load the updated IDF file
        epw_path = r"./ep-model/USA_MT_Charlie.Stanford.720996_TMYx.2007-2021.epw"
        updated_idf = IDF(idf_path, epw_path)

        print(f"Starting simulation with IDF file: {os.path.basename(idf_path)}")
        updated_idf.run(output_prefix='_run', output_suffix='C', readvars=True, output_directory='./ep-model')
            
        # Path to the generated CSV file
        csv_file_path = './ep-model/_runMeter.csv'

        try:
            # Load the generated CSV data
            df = pd.read_csv(csv_file_path)
            
            # Column names
            electricity_col = 'Electricity:Facility [J](Hourly)'
            natural_gas_col = 'NaturalGas:Facility [J](Hourly) '
            
            # Sum the two columns row-wise and convert Joules to kWh (1 kWh = 3.6e6 J)
            df['energy'] = (df[electricity_col] + df[natural_gas_col])  / 3.6e6
            
            self.summed_kwh = df['energy'].iloc[-24:]
            
            # Ensure exactly 24 rows of results
            if len(self.summed_kwh) == 24:
                print("Summed energy values in kWh for rows 2 to 25:")
                print(self.summed_kwh.reset_index(drop=True))  # Display 24 summed rows
            else:
                print(f"Unexpected number of rows in the extracted data: {len(self.summed_kwh)}")

        except Exception as e:
            print(f"An error occurred while processing the CSV file: {e}")
        return self.summed_kwh.tolist()
    
    def _log_iteration(self, iteration, schedule, energy_consumption, loss):
        """Log iteration details including schedule, energy consumption, and loss."""
        log_entry = {
            'iteration': iteration,
            'schedule': schedule.tolist(),
            'energy_consumption': energy_consumption,
            'loss': loss
        }
        
        log_file = f"{self.save_folder}/iteration_log.json"
        
        if os.path.exists(log_file):
            with open(log_file, 'r+') as f:
                log_data = json.load(f)
                log_data.append(log_entry)
                f.seek(0)
                json.dump(log_data, f, indent=4)
        else:
            with open(log_file, 'w') as f:
                json.dump([log_entry], f, indent=4)
        
        print(f"Logged iteration {iteration} details.")