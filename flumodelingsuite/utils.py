### utils.py
# Utility functions


def get_location_codebook():
	'''Retrieve the location codebook as a Pandas DataFrame.'''
	import pandas as pd
	import os, sys
	
	filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "data/location_codebook.csv")
	location_codebook = pd.read_csv(filename)
	
	return location_codebook
    

def convert_location_name_format(value: str, output_format: str) -> str:
	"""
	Convert location name from any valid format to the specified format.

	Available formats:
	"ISO" - Location code in ISO 3166. Countries use ISO 3166-1 alpha-2 country code (e.g. "US") and states/regions use ISO 3166-2 subdivision (e.g. "US-NY").
	"epydemix_population" - Population names used by epydemix (e.g. "United_States", "United_States_New_York").
	"name" - Standard English names (e.g. "United States", "New York").
	"abbreviation" - Two-letter postal abbreviations (e.g. "US", "NY").
	"FIPS" - Two-character Federal Information Processing Standard codes (e.g. "US", "36").
	
	Parameters
	----------
		value (str): The location name to convert, in any valid format.
		output_format (str): The location name format to convert to, either "ISO", "epydemix_population", "name", "abbreviation", or "FIPS".

	Returns
	-------
		str: The converted location name.
	"""
	
	# Retrieve codebook
	codebook = get_location_codebook()

	# Find row with input value
	location = codebook[codebook.isin([value]).any(axis=1)]

	# Ensure value exists
	assert not location.empty, f'Supplied location value {value} does not match any valid format'

	# Match format strings to codebook columns
	format_dict = {'ISO':'ISO', 'epydemix_population':'location_name_epydemix', 'name':'location_name',
				   'abbreviation':'location_abbreviation', 'FIPS':'location_code'}

	# Return location name in requested format
	return location[format_dict[output_format]].values[0]



