#!/usr/bin/python

import helpers
import calculators

if __name__ == "__main__":

    # Parse command line arguments or run initialization flags
    region, control_type, event_type, scaling_option = helpers.load_run_config('./src/config/config.yaml')
    scaling_factors = helpers.get_scaling_factors(scaling_option)
    samples = {}
    base_path = helpers.get_SAMPLES_base_path()
    base_path = helpers.region_and_event_type_check(region, event_type, base_path)

    #Load data filtering by event type, calculate derived variables per event and apply analysis cuts depending on control type
    SAMPLE_PATHS = helpers.get_SAMPLE_PATHS(base_path)
    for sample_name, file_name in SAMPLE_PATHS.items():
        samples[sample_name] = helpers.load_data_filtering_event_type(base_path + file_name, event_type)
        samples[sample_name] = helpers.calculate_derived_variables(samples[sample_name])
        samples[sample_name] = helpers.apply_analysis_cuts(samples[sample_name], control_type)

    samples = helpers.apply_scaling_factors(samples, scaling_factors)
    signal, background = helpers.create_signal_and_background(samples)
    #Preparing the channels list that will be used for yields
    channels = helpers.get_channels_towards_yields(event_type, region)

    # Calculate yields and significances
    yield_results = calculators.calculate_all_yields(samples['DATA'], signal, background, channels)

    # Create binning and plot names and data lists
    variable_binning = helpers.get_VARIABLE_BINNING() 
    plot_data = [samples['DATA']]
    plot_names = ['Data']
    for key in signal.keys():
        plot_data.append(signal[key])
        plot_names.append(f"{key} Signal")
    for key in background.keys():
        plot_data.append(background[key])
        plot_names.append(f"{key} Background") 

    figures = []
    for var, bins in variable_binning.items():
        fig = calculators.plot_distributions(plot_data, plot_names, var, bins, region)
        figures.append(fig)

    # Save results
    calculators.save_results(yield_results, figures, channels, region, control_type, event_type, scaling_option, variable_binning)

    # Print summary
    print("\nAnalysis Summary:")
    print(f"Region: {region}, Control Type: {control_type}, Event Type: {event_type}")
    for channel in channels:
        print(f"Significance for {channel}: {yield_results[channel]['significance']:.2f}")
    
    
    
        
    
