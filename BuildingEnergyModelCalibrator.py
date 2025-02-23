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
    def __init__(self, model_url="http://localhost:11434/", model_name="llama3.2:3b", use_dynamic_temp=True):
        """
        Initialize the model and optimization components.
        
        Args:
            model_url (str): URL for the model API
            model_name (str): Name of the model to use
            use_dynamic_temp (bool): Whether to use dynamic temperature adjustment
        """
        self.model_url = model_url
        self.model_name = model_name
        self.use_dynamic_temp = use_dynamic_temp
        self.chat_history = []
        
        # Initialize with default temperature - will be updated dynamically if enabled
        self._update_model(0.7)

    def get_dynamic_temperature(self, iteration, total_iterations):
        """
        Calculate temperature based on optimization progress.
        
        Args:
            iteration (int): Current iteration
            total_iterations (int): Total number of iterations
            
        Returns:
            float: Temperature value between 0 and 1
        """
        progress = iteration / total_iterations
        if progress < 0.2:
            return 0.7  # Early exploration phase
        elif progress < 0.8:
            return 0.4  # Main optimization phase
        else:
            return 0.2  # Fine-tuning phase

    def _update_model(self, temperature):
        """Update model with new temperature setting."""
        self.model = ChatOllama(
            model=self.model_name,
            base_url=self.model_url,
            temperature=temperature
        )
        self.optimizer_llm_dict = {
            "model_type": "llama3.2",
            "temperature": temperature,
            "top_p": 1,
            "stop": None,
        }

    def run_building_simulation(self, idf_path: str, schedule):
        """Run building simulation with given schedule."""
        # Update the IDF file with the new schedule
        
        schedule_names =  ["BLDG_LIGHT_SCH", "BLDG_OCC_SCH", "BLDG_EQUIP_SCH"]
        self.revise_schedule_compact(idf_path, schedule_names, schedule)
        
        # Run simulation and get results
        updated_idf_path = idf_path[:-4] + "_revised.idf"
        return self.run_simulation(updated_idf_path)
        
    def evaluate_loss(self, simulated_energy_consumption, ground_truth):
        """Calculate loss between simulation result and ground truth."""
        
        loss = np.mean(np.abs(np.array(simulated_energy_consumption) - np.array(ground_truth)))

        print(f"LOSS: {loss:.2f}")
        return loss
    

    # def evaluate_loss(self, simulated_energy_consumption, ground_truth):
    #     """Calculate CVRMSE following ASHRAE Guideline 14-2014."""
        
    #     # Convert inputs to numpy arrays
    #     sim = np.array(simulated_energy_consumption)
    #     gt = np.array(ground_truth)
        
    #     # Calculate the mean of the ground truth values
    #     mean_gt = np.mean(gt)

    #     # Ensure mean_gt is not zero to avoid division errors
    #     if mean_gt == 0:
    #         raise ValueError("Mean of ground truth values is zero, cannot compute CVRMSE.")

    #     # Calculate RMSE using N-1
    #     rmse = np.sqrt(np.sum((sim - gt) ** 2) / (len(gt) - 1))

    #     # Calculate CVRMSE (expressed as a percentage)
    #     cvrmse = round((rmse / mean_gt) * 100, 2)
    #     print(f"CVRMSE: {cvrmse:.2f}%")

    #     return cvrmse

    def load_previous_results(self, results_path):
        """Load previous optimization results from results.json file."""
        if not results_path or not os.path.exists(results_path):
            print(f"No results file found at: {results_path}")
            return set(), None, float('inf')
            
        try:
            with open(results_path, 'r') as f:
                results_data = json.load(f)
            
            # Convert previous pairs into the format used by old_value_pairs_set
            old_value_pairs_set = set()
            for pair in results_data.get('current_pairs', []):
                # Convert list to tuple for set storage
                value_pair = tuple(round(x, 2) for x in pair)
                old_value_pairs_set.add(value_pair)
                
            # Get best schedule and loss
            best_schedule = results_data.get('best_schedule')
            best_loss = results_data.get('best_loss', float('inf'))
            
            if best_schedule is not None:
                best_schedule = np.array(best_schedule)
            
            print(f"Loaded {len(old_value_pairs_set)} previous results from {results_path}")
            print(f"Previous best loss: {best_loss}")
            
            return old_value_pairs_set, best_schedule, best_loss
        except Exception as e:
            print(f"Error loading previous results: {e}")
            return set(), None, float('inf')

    def optimize_parameters(self, base_idf_path: str, ground_truth, num_iterations=50, num_starting_points=5, previous_results_path=None):
        """
        Optimize building parameters to match ground truth.
        
        Args:
            base_idf_path (str): Path to the base IDF file
            ground_truth: Target energy consumption values
            num_iterations (int): Number of optimization iterations
            num_starting_points (int): Number of initial points to try
            previous_results_path (str, optional): Path to previous results.json file to continue from
        """
        # Initialize results storage
        datetime_str = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.save_folder = f"optimization_results_{datetime_str}"
        os.makedirs(self.save_folder, exist_ok=True)
        
        # Initialize or load previous results
        if previous_results_path:
            old_value_pairs_set, best_schedule, best_loss = self.load_previous_results(previous_results_path)
            print("Continuing optimization from previous results")
        else:
            old_value_pairs_set = set()
            best_schedule = None
            best_loss = float('inf')
            print("Starting new optimization")
        
        # Extract base name for simulation files
        base_name = os.path.splitext(os.path.basename(base_idf_path))[0]

        # Generate new starting points if needed
        required_points = num_starting_points - len(old_value_pairs_set) if previous_results_path else num_starting_points
        if required_points > 0:
            print(f"Generating {required_points} new starting points")
            
            # Use high temperature for initial exploration
            if self.use_dynamic_temp:
                self._update_model(0.7)
            
            for start_point in range(required_points):
                unique_idf_name = f"{base_name}_start_point_{start_point}.idf"
                unique_idf_path = os.path.join(self.save_folder, unique_idf_name)
                self._copy_idf_file(base_idf_path, unique_idf_path)
                
                schedule = np.random.uniform(0, 1, 24)
                schedule = np.round(schedule, 2)
                
                energy_consumption = self.run_building_simulation(unique_idf_path, schedule)
                loss = self.evaluate_loss(energy_consumption, ground_truth)
                old_value_pairs_set.add(tuple([round(x, 2) for x in list(schedule)] + energy_consumption + [loss]))
                
                self._log_iteration(f"start_point_{start_point}", schedule, energy_consumption, loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_schedule = schedule
        
        # Optimization loop
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Update temperature if using dynamic adjustment
            if self.use_dynamic_temp:
                temp = self.get_dynamic_temperature(iteration, num_iterations)
                self._update_model(temp)
                print(f"Current temperature: {temp:.2f}")
            
            meta_prompt = self._generate_meta_prompt(old_value_pairs_set, ground_truth)
            new_schedules = self._get_llm_proposals(meta_prompt)
            
            for idx, schedule in enumerate(new_schedules):
                unique_idf_name = f"{base_name}_proposal_{idx}.idf"
                unique_idf_path = os.path.join(self.save_folder, unique_idf_name)
                self._copy_idf_file(base_idf_path, unique_idf_path)
                
                energy_consumption = self.run_building_simulation(unique_idf_path, schedule)
                loss = self.evaluate_loss(energy_consumption, ground_truth)
                old_value_pairs_set.add(tuple([round(x, 2) for x in list(schedule)] + energy_consumption + [loss]))
                
                self._log_iteration(f"iteration_{iteration+1}", schedule, energy_consumption, loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_schedule = schedule
                    print(f"New best loss: {best_loss}")
                
                if best_loss < 5:
                    print(f"Stopping early: best loss {best_loss} is below threshold")
                    self._save_results(self.save_folder, {
                        'iteration': iteration,
                        'best_loss': best_loss,
                        'best_schedule': best_schedule.tolist() if best_schedule is not None else None,
                        'current_pairs': list(old_value_pairs_set),
                        'final_temperature': self.optimizer_llm_dict['temperature']
                    })
                    return best_schedule, best_loss
            
            self._save_results(self.save_folder, {
                'iteration': iteration,
                'best_loss': best_loss,
                'best_schedule': best_schedule.tolist() if best_schedule is not None else None,
                'current_pairs': list(old_value_pairs_set),
                'current_temperature': self.optimizer_llm_dict['temperature']
            })
        
        return best_schedule, best_loss

    def _copy_idf_file(self, source_path: str, destination_path: str):
        """Helper method to copy IDF file while ensuring the destination directory exists."""
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            with open(source_path, 'r') as source, open(destination_path, 'w') as dest:
                dest.write(source.read())
            print(f"Successfully copied IDF file to: {destination_path}")
        except Exception as e:
            print(f"Error copying IDF file: {e}")
            raise

    # def _generate_meta_prompt(self, old_value_pairs_set, ground_truth):
    #     """Generate meta prompt for LLM."""
    #     pairs_str = "\n".join([
    #         f"schedule: {list(pair[:-1])}\nloss: {pair[-1]}"
    #         for pair in sorted(old_value_pairs_set, key=lambda x: x[-1])[:10]
    #     ])
        
    #     return f"""
    #     Help me optimize a 24-hour building occupancy schedule to match a target energy consumption pattern.
    #     The schedule should contain 24 values between 0 and 1 representing hourly occupancy rates.
        
    #     Previous attempts and their losses:
    #     {pairs_str}
        
    #     Target energy consumption pattern:
    #     {ground_truth}
        
    #     Propose a new schedule that might achieve lower loss. Output should be exactly 24 numbers between 0 and 1.
    #     Consider typical building occupancy patterns (e.g., higher during work hours, lower at night).

    #     Higher occupancy typically increases heating, cooling, and ventilation demands, while unoccupied periods allow for energy-saving. Dynamic schedules, such as flexible work hours or intermittent use, create variability in energy demand. Accurate occupancy modeling is crucial for optimizing energy efficiency, as overestimating leads to wasted energy, while underestimating compromises comfort and performance.

    #     """

    def _generate_meta_prompt(self, old_value_pairs_set, ground_truth):
        """Generate meta prompt for LLM with structured previous attempts data."""
        # Convert the set of tuples into list of dictionaries
        previous_attempts = []
        for pair in sorted(old_value_pairs_set, key=lambda x: x[-1])[:10]:  # Get top 10 best attempts
            # Split the tuple into its components:
            # First 24 values are schedule
            # Next 24 values are energy consumption
            # Last value is loss
            schedule = list(pair[:24])
            energy_consumption = list(pair[24:-1])
            loss = pair[-1]
            
            attempt_dict = {
                "schedule": schedule,
                "energy_consumption": energy_consumption,
                "loss": loss
            }
            previous_attempts.append(attempt_dict)
        
        # Format the previous attempts for the prompt
        attempts_str = ""
        for i, attempt in enumerate(previous_attempts, 1):
            attempts_str += f"\nAttempt {i}:\n"
            attempts_str += f"Schedule: {attempt['schedule']}\n"
            attempts_str += f"Energy Consumption (kWh): {attempt['energy_consumption']}\n"
            attempts_str += f"Loss (CVRMSE): {attempt['loss']:.2f}%\n"
        
        return f"""
        Help me optimize a 24-hour building occupancy schedule to match a target energy consumption pattern.
        The schedule should contain 24 values between 0 and 1 representing hourly occupancy rates.
        
        Previous attempts and their results:{attempts_str}
        
        Target energy consumption pattern (kWh):
        {ground_truth}
        
        Please analyze the previous attempts and propose a new schedule that might achieve lower loss, considering:
        1. The relationship between schedules and their resulting energy consumption
        2. Patterns in more successful attempts (lower loss values)
        3. How schedule changes affect energy consumption throughout the day
        4. Building energy usage characteristics:
        - Higher occupancy typically increases HVAC and lighting demands
        - Unoccupied periods allow for energy savings
        - Dynamic schedules can create variable energy demand patterns
        
        Output should be exactly 24 numbers between 0 and 1 (with up to two decimal points), representing hourly occupancy rates.
        Focus on creating realistic occupancy patterns (e.g., higher during work hours, lower at night)
        while optimizing to match the target energy consumption pattern.
        
        Note: Compare how the energy consumption values differ from the target at each hour and adjust
        the schedule accordingly to minimize these differences.
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
                    schedule = np.round(np.clip(numbers[:24], 0, 1), 2)
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
        plt.plot(range(24), self.run_building_simulation(idf_path, best_schedule), 'r--', 
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
        updated_idf_path = idf_path[:-4] + "_revised.idf"
        self.idf.saveas(updated_idf_path)
        print(f"Updated IDF file saved at {updated_idf_path}")

    def revise_schedule_interval(self, idf_path: str, schedule_names: list, hourly_values: list):
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
        schedules = self.idf.idfobjects['SCHEDULE:DAY:INTERVAL']
        for schedule_name in schedule_names:
            for schedule in schedules:
                if schedule.Name == schedule_name:
                    # Generate the "Until" time fields
                    until_times = [f"{i + 1:02}:00" for i in range(24)]

                    # Update the fields in Schedule:Compact
                    for i in range(24):
                        until_field = f"Time_{(i + 1)}"  # Alternating "Until" fields
                        value_field = f"Value_Until_Time_{(i + 1)}"  # Corresponding value fields

                        # Format values properly
                        formatted_value = f"{hourly_values[i]:.2f}".lstrip('0') if hourly_values[i] != 0 else "0.0"

                        setattr(schedule, until_field, until_times[i])  # Set "Until" time
                        setattr(schedule, value_field, formatted_value)  # Set value

                    print(f"Updated Schedule:Day:Interval {schedule_name} with new hourly values.")
                    break
            else:
                print(f"Schedule:Day:Interval with name '{schedule_name}' not found.")
                continue

        # Save the updated IDF
        updated_idf_path = idf_path[:-4] + "_revised.idf"
        self.idf.saveas(updated_idf_path)
        print(f"Updated IDF file saved at {updated_idf_path}")

    def run_simulation(self, idf_path: str):
        """Run the simulation using the updated IDF file and process the generated energy data."""
        # Load the updated IDF file
        epw_path = r"./ep-model/stanford_mountain_view.epw"
        updated_idf = IDF(idf_path, epw_path)

        print(f"Starting simulation with IDF file: {os.path.basename(idf_path)}")
        file_name = os.path.basename(idf_path)[:-4]

        updated_idf.run(output_prefix=file_name, output_suffix='C', readvars=True, output_directory='./ep-model')
            
        # Path to the generated CSV file
        csv_file_path = './ep-model/' + file_name + "Meter.csv"

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
                
                print(self.summed_kwh.reset_index(drop=True)) 
            else:
                print(f"Unexpected number of rows in the extracted data: {len(self.summed_kwh)}")

        except Exception as e:
            print(f"An error occurred while processing the CSV file: {e}")
        return [round(x, 2) for x in self.summed_kwh.tolist()]
    
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