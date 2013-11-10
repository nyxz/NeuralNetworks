// Neuron network implementation. Backpropagation.

var Connection = function() {
	// weights
	this.weight = random_weight();
	this.delta_weight = 1;
}

var Neuron = function(layer_num, neuron_num, output_size) {
	// meta data
	this.layer_num = layer_num;
	this.neuron_num = neuron_num;

	// vector of connections - the weights between the neurons
	this.output_weights = [];

	// for backpropagation
	this.gradient;
	this.output;
	this.alpha = 0.4;
	this.eta = 0.2;

	// number of output to create for the next layer
	this.output_size = output_size;


	this.calc_output_weights = function() {
		this.output_weights = [];
		for (var i = 0; i < this.output_size; i++) {
			this.output_weights.push(new Connection());
		}
	}
	this.calc_output_weights();


	/*
	Sum all the outputs from the previous layer (the inputs to the curr layer)
	Include the bias node from the previous layer 
	*/
	this.feed_forward = function(prev_layer) {
		var sum = 0.0;

		for (var curr_neuron = 0; curr_neuron < prev_layer.length; curr_neuron++) {
			var curr_neuron_weight = prev_layer[curr_neuron].output_weights[this.neuron_num].weight;
			sum += to_fixed_precision(prev_layer[curr_neuron].output * curr_neuron_weight);
		}

		this.output = activation_function(sum);
	}

	this.calculate_gradient = function(target) {
		this.gradient = (target - this.output) * activation_function_derivative(this.output);
	}

	this.delta_sum = function(next_layer) {
		var sum = 0.0;
		for (var n = 0; n < next_layer.length - 1; n++) { // without the bias
			sum += this.output_weights[n].weight * next_layer[n].gradient;
		}
		return sum;
	}

	this.calc_hidden_gradient = function(next_layer) {
		var d_sum = this.delta_sum(next_layer);
		this.gradient = d_sum * activation_function_derivative(this.output);
	}

	this.update_weight = function(prev_layer) {
		for (var n = 0; n < prev_layer.length; n++) {
			var neuron = prev_layer[n];
			var old_delta_weight = to_fixed_precision(neuron.output_weights[this.neuron_num].delta_weight);
			var new_delta_weight = to_fixed_precision((this.eta * neuron.output * this.gradient) + (this.alpha * old_delta_weight));

			neuron.output_weights[this.neuron_num].delta_weight = new_delta_weight;
			neuron.output_weights[this.neuron_num].weight += new_delta_weight;
		}
	}
}

// Network class
var Network = function(topology) {
	
	// layer[layer_num][neuron_num]
	this.layers = []; // vector of neurons
	this.topology = topology;
	this.error = 0.0;
	this.output_gradient = [];

	this.feed_forward = function(input_values) {
		if (input_values.length != (this.layers[0].length - 1)) { // -1 because of the BIAS
			alert("Input values and input neurons number is different!");
			return;
		}

		// set bias values to 1
		for (var i = 0; i < this.layers.length; i++) {
			this.layers[i][this.layers[i].length - 1].output = 1;
		}

		// fill input neurons
		for (var i = 0; i < input_values.length; i++) {
			this.layers[0][i].output = input_values[i];
		}

		// Forward propagate
		for (var i = 1; i < this.layers.length; i++) { // start from 1 because we already init the input layer
			var prev_layer = this.layers[i - 1];
			for (var n = 0; n < (this.layers[i].length - 1); n++) { // -1 because of the bias
				this.layers[i][n].feed_forward(prev_layer);
			}
		}
	}
	
	this.back_propagation = function(target_values) {
		var output_layer = this.layers[this.layers.length - 1];

		// calcutate the error at the end
		var old_err = this.error;
		this.error = calc_error_RMS(target_values, output_layer);

		this.avg_err = (this.avg_err + this.error) / this.process_nums 

		// calculate gradients for all output layer neurons
		for (var i = 0;  i < (output_layer.length - 1); i++) { // -1 to remove the bias
			output_layer[i].calculate_gradient(target_values[i]);
		}

		// calculate gradients for all hidden layer neurons 
		// (layer.size() - 2 is the first hidden layer backwards)
		for (var i = this.layers.length - 2; i > 0; i--) { // -2 to point the last hidden layer
			var hidden_layer = this.layers[i];
			var next_layer = this.layers[i + 1];

			for (var n = 0; n < hidden_layer.length; n++) {
				hidden_layer[n].calc_hidden_gradient(next_layer);
			}
		}

		// compute deltas for the all weights using eta
		// add delta to each weight 
		for (var i = this.layers.length - 1; i > 0; i--) { // -2 to point the last hidden layer
			var layer = this.layers[i];
			var prev_layer = this.layers[i - 1];

			for (var n = 0; n < layer.length - 1; n++) { // -1 to remove the bias
				layer[n].update_weight(prev_layer);
			}
		}
	}
	
	this.show_results = function() {
		console.log(this.layers[this.layers.length - 1]);
		for (var i = 0; i < this.layers[this.layers.length - 1].length - 1; i++) {
			console.log(this.layers[this.layers.length - 1][i].output);
		}
	}

	this._init_layers = function() {

		for (var cur_layer = 0; cur_layer < this.topology.length; cur_layer++) {
			var num_outputs = (cur_layer == (this.topology.length - 1)) ? 0 : this.topology[cur_layer + 1]; 

			for (var cur_neuron = 0; cur_neuron <= this.topology[cur_layer]; cur_neuron++) { // "<=" - this is due to the BIAS we add
				if (this.layers[cur_layer] == undefined) {
					this.layers[cur_layer] = [];
				}
				this.layers[cur_layer].push(new Neuron(cur_layer, cur_neuron, num_outputs));
			}
		}
	}

	this._init_layers();
}

var main = function() {
	var topology = [5, 3, 10, 1]; // ex. [1, 2, 3]
	myNet = new Network(topology);

	var input_values = [1, 2, 3, 4, 5];
	myNet.feed_forward(input_values);

	var target_values = [5];
	myNet.back_propagation(target_values);

	myNet.show_results();
}