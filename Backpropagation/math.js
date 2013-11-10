var PRECISION = 5;

var random_weight = function() {
	return parseFloat(Math.random().toFixed(PRECISION));
}

var activation_function = function(x) {
	// hiperbolic tangent - a.k.a tanh
	return to_fixed_precision(Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

var activation_function_derivative = function(x) {
	return to_fixed_precision(1 - activation_function(x)) * (1 + activation_function(x));
}

var to_fixed_precision = function(x) {
	return parseFloat(x.toFixed(PRECISION));
}

var calc_error_RMS = function(targets, outputs) {
	var sum = 0.0;
	for (var i = 0; i < targets.length; i++) {
		var delta = targets[i] - outputs[i].output;
		sum += delta * delta;
	}

	return Math.sqrt(sum / outputs.length);
}