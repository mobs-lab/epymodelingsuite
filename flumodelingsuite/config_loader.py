### config_loader.py
# Functions for loading and validating configuration files (defined in YAML format).

from typing import Any
import logging

logger = logging.getLogger(__name__)

from epydemix.model import EpiModel

import numpy as np
import scipy

import ast
import operator

# === Utility functions for evaluating expressions ===
# Allowed binary operators mapping
_allowed_operators = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.Div: operator.truediv,
	ast.Pow: operator.pow,
	ast.Mod: operator.mod,
}

# Allowed unary operators mapping
_allowed_unary_operators = {
	ast.UAdd: operator.pos,
	ast.USub: operator.neg,
}

# Names of top‐level modules we allow
_allowed_modules = {'np', 'scipy'}

class SafeEvalVisitor(ast.NodeVisitor):
	"""A NodeVisitor that only allows numeric, numpy, and scipy expressions."""

	def visit(self, node):
		t = type(node)
		# Permit only these node types
		if t in (ast.Expression, ast.BinOp, ast.UnaryOp,
				 ast.Constant, ast.Num, ast.Load,
				 ast.Name, ast.Attribute, ast.Call):
			return super().visit(node)
		raise ValueError(f"Disallowed expression: {t.__name__}")

	def visit_BinOp(self, node):
		self.visit(node.left)
		self.visit(node.right)
		if type(node.op) not in _allowed_operators:
			raise ValueError(f"Operator {type(node.op).__name__} not allowed")

	def visit_UnaryOp(self, node):
		self.visit(node.operand)
		if type(node.op) not in _allowed_unary_operators:
			raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")

	def visit_Constant(self, node):
		# Only allow numeric constants
		if not isinstance(node.value, (int, float)):
			raise ValueError(f"Constant of type {type(node.value).__name__} not allowed")

	def visit_Num(self, node):
		# For Python <3.8
		if not isinstance(node.n, (int, float)):
			raise ValueError(f"Num of type {type(node.n).__name__} not allowed")

	def visit_Name(self, node):
		# Only allow top‐level names 'np' and 'scipy'
		if node.id not in _allowed_modules:
			raise ValueError(f"Name '{node.id}' is not allowed")

	def visit_Attribute(self, node):
		# Recursively ensure base is allowed module (np or scipy)
		if self._is_allowed_attr_chain(node):
			# visit the base value to enforce nested checks
			self.visit(node.value)
		else:
			raise ValueError(f"Attribute access '{ast.dump(node)}' not allowed")

	def _is_allowed_attr_chain(self, node):
		# Base case: node.value is Name in allowed_modules
		if isinstance(node.value, ast.Name) and node.value.id in _allowed_modules:
			return True
		# Recursive: node.value is another Attribute
		if isinstance(node.value, ast.Attribute):
			return self._is_allowed_attr_chain(node.value)
		return False

	def visit_Call(self, node):
		# Only allow calls of form (np.xxx(...)) or (scipy.xxx(...))
		if isinstance(node.func, ast.Attribute):
			# validate the attribute chain (np or scipy)
			self.visit(node.func)
			# validate all arguments
			for arg in node.args:
				self.visit(arg)
			for kw in node.keywords:
				self.visit(kw.value)
		else:
			raise ValueError(f"Function calls other than np.xxx or scipy.xxx are not allowed")

def _safe_eval(expr: str) -> Any:
	"""
	Safely evaluate a numeric expression from a string, allowing literal numbers,
	basic arithmetic operators, and functions from numpy and scipy.

	Parameters
	----------
	expr : str
		The expression to evaluate (e.g. "1/10" or "np.exp(-2) + 3 * np.sqrt(4)").

	Returns
	-------
	Any
		The result of evaluating the expression. Depending on the expression, this
		may be one of:
		  - A Python numeric type: int, float, or complex.
		  - A NumPy scalar (e.g. numpy.int64, numpy.float64).
		  - A NumPy ndarray.
		  - A SciPy sparse matrix (subclass of scipy.sparse.spmatrix).

	"""
	# Parse into an AST
	tree = ast.parse(expr, mode='eval')

	# Validate AST nodes
	SafeEvalVisitor().visit(tree)
	
	# Compile and evaluate with restricted globals
	code = compile(tree, filename="<safe_eval>", mode="eval")
	return eval(code, {'__builtins__': None, 'np': np, 'scipy': scipy}, {})

# === Model setup functions ===
def _add_model_compartments_from_config(model, config):
	"""
	Add compartments to the EpiModel instance from the configuration dictionary.
	
	Parameters
	----------
		model (EpiModel): The EpiModel instance to which compartments will be added.
		config (dict): Configuration dictionary containing compartment definitions.
		
	Returns
	----------
		EpiModel: EpiModel instance with compartments added.
	"""
	
	if 'compartments' not in config['model']:
		return model
	
	# Add compartments to the model
	try:
		compartment_ids = [ compartment['id'] for compartment in config['model']['compartments'] ]
		model.add_compartments(compartment_ids)
		logger.info(f"Added compartments: {compartment_ids}")
	except Exception as e:
		raise ValueError(f"Error adding compartments: {e}")

	return model

def _add_model_transitions_from_config(model, config):
	"""
	Add transitions between compartments to the EpiModel instance from the configuration dictionary.

	Parameters
	----------
		model (EpiModel): The EpiModel instance to which compartment transitions will be added.
		config (dict): Configuration dictionary containing compartment transitions.

	Returns
	----------
		EpiModel: EpiModel instance with compartment transitions added.
	"""

	if 'transitions' not in config['model']:
		return model

	# Add transitions to the model
	for transition in config['model']['transitions']:
		if transition['type'] == "mediated":
			try:
				model.add_transition(
					transition['source'],
					transition['target'],
					params=(
						transition['mediators']['rate'],
						transition['mediators']['source']
					),
					kind=transition['type']
				)
				logger.info(f"Added mediated transition: {transition['source']} -> {transition['target']} (mediator: {transition['mediators']['source']}, rate: {transition['mediators']['rate']})")
			except Exception as e:
				raise ValueError(f"Error adding mediated transition {transition}: {e}")
		elif transition['type'] == "spontaneous":
			try:
				model.add_transition(
					transition['source'],
					transition['target'],
					params=transition['rate'],
					kind=transition['type']
				)
				logger.info(f"Added spontaneous transition: {transition['source']} -> {transition['target']} (rate: {transition['rate']})")
			except Exception as e:
				raise ValueError(f"Error adding spontaneous transition {transition}: {e}")

	return model

def _add_model_parameters_from_config(model, config):
	"""
	Add parameters to the EpiModel instance from the configuration dictionary.
	
	Parameters
	----------
		model (EpiModel): The EpiModel instance to which parameters will be added.
		config (dict): Configuration dictionary containing model parameters.
	
	Returns:
	----------
		EpiModel: EpiModel instance with parameters added.
	"""
	
	if 'parameters' not in config['model']:
		return model

	# Add parameters to the model
	parameters_dict = {}
	for key, data in config['model']['parameters'].items():
		if data['type'] == 'constant':
			parameters_dict[key] = data['value']
		elif data['type'] == 'array':
			parameters_dict[key] = data['values']
		elif data['type'] == 'expression':
			parameters_dict[key] = _safe_eval(data['value'])
	
	try:
		model.add_parameter(parameters_dict=parameters_dict)
		logger.info(f"Added parameters: {list(parameters_dict.keys())}")
	except Exception as e:
		raise ValueError(f"Error adding parameters to model: {e}")

	return model

def _set_population_from_config(model, config):
	"""
	Set the population for the EpiModel instance from the configuration dictionary.
	
	Parameters
	----------
		model (EpiModel): The EpiModel instance for which the population will be set.
		config (dict): Configuration dictionary containing population details.

	Returns
	----------
		EpiModel: EpiModel instance with the population set.
	"""

	from epydemix.population import load_epydemix_population

	if 'population' not in config['model']['simulation']:
		return model

	try:
		population_name = config.get('model').get('simulation').get('population')
		population = load_epydemix_population(population_name)
		model.set_population(population)
		logger.info(f"Model population set to: {population_name}")
	except Exception as e:
		raise ValueError(f"Error setting population: {e}")

	return model

def _add_school_closure_intervention_from_config(model, config):
	"""
	Apply a school closure intervention to the EpiModel instance.
	
	Parameters
	----------
		model (EpiModel): The EpiModel instance to which the intervention will be applied.
		config (dict): Configuration dictionary containing intervention details.

	Returns
	----------
		EpiModel: EpiModel instance with the intervention applied.
	"""

	# Check if interventions are defined in the config
	if 'interventions' not in config['model']:
		return model
	
	# Load school closure functions
	from .school_closures import make_school_closure_dict, add_school_closure_interventions

	# Extract school closure interventions
	school_closures_interventions = [intervention for intervention in config['model']['interventions'] if intervention['type'] == 'school_closure']

	# Confirm that there are school closure interventions to apply
	if len(school_closures_interventions) == 0:
		return model

	for intervention in school_closures_interventions:
		try:
			closure_dict = make_school_closure_dict(intervention['years'])
			add_school_closure_interventions(
					model=model,
					closure_dict=closure_dict,
					reduction_factor=intervention['reduction_factor']
			)
			logger.info(f"Applied school closure intervention for years: {intervention['years']} with reduction factor: {intervention['reduction_factor']}")
		except Exception as e:
			raise ValueError(f"Error applying school closure intervention {intervention}:\n{e}")

	return model

def setup_epimodel_from_config(config):
	"""
	Set up an EpiModel instance from a configuration dictionary.
	
	Parameters
	----------
		config (dict): Configuration dictionary containing model details. Usually loaded from a YAML file.

	Returns
	----------
		EpiModel: An instance of EpiModel configured according to the provided settings.
	"""
	
	# Validate that 'model' key exists in config
	if 'model' not in config:
		raise ValueError("Configuration must contain a 'model' key.")

	# Create an empty instance of EpiModel
	model = EpiModel()

	# Set the model name if provided in the config
	if 'name' in config['model']:
		model.name = config['model']['name']
	
	# Set population
	model = _set_population_from_config(model, config)

	# Set up compartments
	model = _add_model_compartments_from_config(model, config)

	# Set up transitions
	model = _add_model_transitions_from_config(model, config)

	# Set up parameters
	model = _add_model_parameters_from_config(model, config)

	# Apply school closure
	model = _add_school_closure_intervention_from_config(model, config)

	return model