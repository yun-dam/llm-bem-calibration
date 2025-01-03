from BuildingParameterEstimator import BuildingParameterEstimator 

def main():
    # Initialize the estimator
    estimator = BuildingParameterEstimator()

    # Step 1: Provide initial building information
    building_info = """
    Building name: DOE Office building
    Year of construction: 2011
    Building usetype: Office
    """
    print("Generating initial occupancy estimation...")
    initial_output = estimator.create_estimation(building_info)
    print("\nInitial Output:")
    print(initial_output)

    # Step 2: Refine estimation based on user feedback
    print("\nPlease provide details about the building's occupancy pattern (e.g., timings, peak hours):")
    user_response = input("\nYour input:\n")

    ## Example: On weekdays, this office building typically sees a gradual increase in occupancy starting from 7:00 AM as employees begin arriving for work. By 9:00 AM, most employees have settled in, and occupancy reaches its peak. This high level of activity continues through the morning until around noon. During lunch hours, between 12:00 PM and 1:00 PM, there is a slight decrease in occupancy as employees leave for breaks or meals. However, the building quickly returns to near-peak levels of occupancy in the afternoon, from 1:00 PM to 5:00 PM, as work resumes. After 5:00 PM, occupancy begins to decline as employees leave for the day, with most departures occurring by 7:00 PM. Beyond this time, the building is largely unoccupied, except for cleaning crews, security staff, or occasional employees working late. 
    
    
    print("\nRefining estimation based on user input...")
    refined_output, chat_history = estimator.refine_estimation(user_response)
    print("\nRefined Output:")
    print(refined_output)

    # Step 3: Validate the final output
    print("\nValidating the output...")
    validated_output = estimator.validate_output(refined_output)
    print("\nValidated Output:")
    print(validated_output)

    # Optional: Print chat history
    # print("\nChat History:")
    # for message in chat_history:
    #     print(message.content)

if __name__ == "__main__":
    main()
