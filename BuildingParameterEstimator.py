from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from eppy.modeleditor import IDF


class BuildingParameterEstimator:
    def __init__(self, model_url="http://localhost:11434/", model_name="llama3.2:3b"):
        """Initialize the model and other required components."""
        self.model = ChatOllama(model=model_name, base_url=model_url)
        self.chat_history = []
        self._initialize_prompts()
        self._initialize_output_parser()


    def _initialize_prompts(self):

        """Initialize prompt templates."""

        self.estimate_template = """\
        You are an expert building occupancy analyzer. Your task is to estimate hourly occupancy schedules (0=vacant to 1=full capacity) based on building descriptions.

        Building Description: {text}

        current_estimation: Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma-separated Python list of 24 estimation samples from 0 hour to 23 hour. Ensure that the estimated values align with realistic occupancy patterns.

        following_question: If you need more information to estimate occupancy schedule more accurately, ask user relevant questions.

        validation_check: A brief explanation of how the estimation aligns with typical patterns, or why it deviates.

        Every time you respond, format the output in every conversation as JSON with the following keys:

          "current_estimation": Python list of 24 values,
          "following_question": "Your next question, or null if no further questions are needed",
          "validation_check": "A brief explanation of how the estimation aligns with typical patterns, or why it deviates."

        CRITICAL RULES:
        1. current_estimation MUST:
        - Contain EXACTLY 24 numbers
        - Each number MUST be between 0 and 1
        - Values represent hours 0 (midnight) to 23 (11 PM)
        - Follow realistic patterns:
            * Office: Peak 9 AM - 5 PM
            * Residential: Higher evenings/mornings
            * Retail: Peak afternoon/evening
        - For example, for an office building with business hours from 8 AM to 6 PM 
        : [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.9, 1.0, 1.0, 0.9, 0.8, 1.0, 1.0, 0.9, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],

        2. following_question MUST:
        - Start with these questions first:
            "What are typical operation hours?"
            "When is the building fully/minimally occupied?"
        - Be specific and focused on occupancy patterns
        - Be null if no more questions needed

        3. validation_check MUST:
        - Explain why the estimation makes sense
        - Note any assumptions made
        - Highlight any unusual patterns

        VALIDATION STEPS:
        Before responding, verify:
        1. Are there exactly 24 values?
        2. Is each value between 0-1?
        3. Do the values follow realistic patterns?
        4. Is the JSON format exact?

        Ask these clarifying questions first:
        - "What are the typical hours of operation or occupancy?"
        - "Are there any specific times when the building is fully or minimally occupied?"

        Building information: {text}

        {format_instructions}
        """
        self.user_template = """\
        
        {text}

        Remember to maintain exact JSON format and the key "current_estimation" MUST have 24 hourly occupancy values between 0 and 1.

        """

    def _initialize_output_parser(self):
        """Initialize the structured output parser."""
        self.current_estimation_schema = ResponseSchema(
            name="current_estimation",
            description="Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma-separated Python list of 24 estimation samples from 0 hour to 23 hour."
        )
        self.following_question_schema = ResponseSchema(
            name="following_question",
            description="If you need more information to estimate occupancy schedule more accurately, ask user relevant questions. This includes the first question."
        )
        self.validation_check_schema = ResponseSchema(
            name="validation_check",
            description="A brief explanation of how the estimation aligns with user's description on occupancy ."
        )
        self.output_parser = StructuredOutputParser.from_response_schemas([
            self.current_estimation_schema,
            self.following_question_schema,
            self.validation_check_schema
        ])
        self.format_instructions = self.output_parser.get_format_instructions()

    def create_estimation(self, building_info: str):
        """
        Generate the first estimation using default values and initial questions.
        Uses the default office building schedule pattern from the template.
        """
        # Default estimation for an office building as specified in the template
        default_estimation = [0, 0, 0, 0, 0, 0, 0.1, 0.8, 1, 1, 1, 1, 1, 1, 0.9, 0.7, 0.3, 0.1, 0, 0, 0, 0, 0, 0]
        
        # Initial questions as specified in the template
        initial_questions = [
            "What are the typical hours of operation or occupancy?",
            "Are there any specific times when the building is fully or minimally occupied?"
        ]
        
        # Format the questions into a single string
        questions_formatted = "\n".join([f"- {q}" for q in initial_questions])
        
        # Create the initial output dictionary
        output_dict = {
            "current_estimation": default_estimation,
            "following_question": questions_formatted,
            "validation_check": "Initial estimation based on typical office building pattern: "
                            "minimal occupancy during night hours (0-6), "
                            "gradual increase during morning arrival (7-8), "
                            "peak occupancy during business hours (9-14), "
                            "gradual decrease during afternoon/evening departure (15-17), "
                            "and minimal occupancy during night (18-23)."
        }
        
        # Save system message to chat history
        self.system_prompt = SystemMessagePromptTemplate.from_template(template=self.estimate_template)
        self.system_messages = self.system_prompt.format_messages(
            text=building_info,
            format_instructions=self.format_instructions
        )
        self.chat_history.extend(self.system_messages)
        
        return output_dict

    

    def refine_estimation(self, user_response: str):
        """Refine the estimation based on user responses with error handling and re-estimation."""
        
        human_prompt = HumanMessagePromptTemplate.from_template(template=self.user_template)
        human_messages = human_prompt.format_messages(text=user_response)
        
        # Append user input to chat history
        self.chat_history.append(human_messages[-1])

        # Generate AI messages based on the latest user input
        # ai_prompt = AIMessagePromptTemplate.from_template(template=self.chat_history[-1].content)
        # ai_messages = ai_prompt.format_messages()
        messages_combined = self.chat_history

        # Invoke the model and parse the response
        response_combined = self.model.invoke(messages_combined)

        output_dict_combined = self.output_parser.parse(response_combined.content)
        
        # Append the new question to chat history and return the output
        
        # self.chat_history.append(AIMessage(content=output_dict_combined.get("following_question")))

        return output_dict_combined, self.chat_history

    def revise_schedule_compact(self, idf_path: str, schedule_name: str, hourly_values: list):
        """
        Revises a Schedule:Compact object in the IDF file with the provided hourly values.

        Parameters:
        - idf_path: Path to the IDF file.
        - schedule_name: Name of the Schedule:Compact object to modify.
        - hourly_values: List of 24 float values representing hourly schedule.
        """
        # Validate input length
        if len(hourly_values) != 24:
            raise ValueError("Hourly values list must contain exactly 24 values.")

        # Load the IDF file
        IDF.setiddname("./ep-model/Energy+.idd")  
        self.idf = IDF(idf_path)

        # Find the Schedule:Compact object
        schedules = self.idf.idfobjects['SCHEDULE:COMPACT']
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
            return

        # Save the updated IDF
        updated_idf_path = "./ep-model/updated_ep.idf"
        self.idf.saveas(updated_idf_path)
        print(f"Updated IDF file saved at {updated_idf_path}")

    def run_simulation(self, idf_path: str):
        """Run the simulation using the updated IDF file."""
        # Load the updated IDF file 
        
        epw_path = "./ep-model/USA_MT_Charlie.Stanford.720996_TMYx.2007-2021.epw"
        updated_idf = IDF(idf_path, epw_path)
        updated_idf.run(output_prefix = '_run', output_suffix = 'L', readvars=True)


    def validate_output(self, output_dict):
        """Validate and fix the output if needed."""
        if not output_dict.get("current_estimation"):
            output_dict["current_estimation"] = [0] * 24
        
        est = output_dict["current_estimation"]
        if len(est) > 24:
            output_dict["current_estimation"] = est[:24]
        elif len(est) < 24:
            output_dict["current_estimation"] = est + [0] * (24 - len(est))
        
        # Ensure values are between 0 and 1
        output_dict["current_estimation"] = [min(max(float(x), 0), 1) for x in output_dict["current_estimation"]]
        
        return output_dict