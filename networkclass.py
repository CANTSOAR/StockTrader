import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import copy


class Model:

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss_function = loss
        
        if optimizer is not None:
            self.optimizer = optimizer
        
        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, X, y, *, epochs, printevery, validation_data=None, batch_size=None):
        self.accuracy.init(y)

        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size + bool(len(X) % batch_size) * 1

        for epoch in range(1, epochs+1):

            if train_steps != 1: print("Epoch:", epoch)

            self.loss_function.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training = True)

                data_loss, regularization_loss = self.loss_function.calculate(output, batch_y, include_reg_loss=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)    
                self.optimizer.post_update_params()

                if (not step % printevery or step == train_steps - 1) and train_steps != 1:
                    print("Step:", step, "Loss:", loss, "(Data Loss:", data_loss, "Reg Loss:", regularization_loss, ") Avg Output:", np.average(predictions), "Acc:", accuracy, "LR:", self.optimizer.current_learning_rate)

            epoch_data_loss, epoch_reg_loss = self.loss_function.calculate_accumulated(include_reg_loss = True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if train_steps != 1: print("Epoch Loss:", epoch_loss, "(Epoch Data Loss:", epoch_data_loss, "Epoch Reg Loss:", epoch_reg_loss, ") Avg Output:", np.average(predictions), "Acc:", epoch_accuracy, "LR:", self.optimizer.current_learning_rate)
            elif not epoch % printevery: print("Epoch", epoch, "Epoch Loss:", epoch_loss, "(Epoch Data Loss:", epoch_data_loss, "Epoch Reg Loss:", epoch_reg_loss, ") Avg Output:", np.average(predictions), "Acc:", epoch_accuracy, "LR:", self.optimizer.current_learning_rate)

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size = batch_size)
            

    def finalize(self):
        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:   
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss_function
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        if self.loss_function is not None:
            self.loss_function.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss_function, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxActivation_CategoricalCrossEntropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss_function.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        print("Validating...")

        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size + bool(len(X_val) % batch_size) * 1

        self.loss_function.new_pass()
        self.accuracy.new_pass()

        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size + bool(len(X_val) % batch_size) * 1

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training = False)

            self.loss_function.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss_function.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print("Validation: {Loss:", validation_loss, "Acc:", validation_accuracy, "}")

    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):
        for layer, paramater_set in zip(self.trainable_layers, parameters):
            layer.set_parameters(*paramater_set)

    def save_parameters(self, path):
        print("Saving Parameters...")

        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        print("Loading Parameters...")

        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        print("Saving Model...")

        model = copy.deepcopy(self)

        model.loss_function.new_pass()
        model.accuracy.new_pass()

        model.loss_function.__dict__.pop("output", None)
        model.accuracy.__dict__.pop("dinputs", None)

        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)

        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        print("Loading Model...")

        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    def predict(self, X, *, batch_size = None):
        print("Predictiing...")

        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size + bool(len(X) % batch_size) * 1

        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            batch_output = self.forward(batch_X, training = False)

            output.append(batch_output)

        return np.vstack(output)

class Layer_Input:

    def forward(self, inputs, training):
        self.output = inputs


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regulizer_l1=0, weight_regulizer_l2=0, bias_regulizer_l1=0, bias_regulizer_l2=0):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regulizer_l1 = weight_regulizer_l1
        self.weight_regulizer_l2 = weight_regulizer_l2
        self.bias_regulizer_l1 = bias_regulizer_l1
        self.bias_regulizer_l2 = bias_regulizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regulizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights = self.weight_regulizer_l1 * dl1


        if self.weight_regulizer_l2 > 0:
            self.dweights += 2 * self.weight_regulizer_l2 * self.weights


        if self.bias_regulizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases = self.bias_regulizer_l1 * dl1


        if self.bias_regulizer_l2 > 0:
            self.dbiases += 2 * self.bias_regulizer_l2 * self.biases

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Activation_Softmax:

    def forward(self, inputs, training):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)

    def predictions(self, outputs):
        return (outputs > .5) * 1


class Activation_Linear:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs


class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_reg_loss=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_reg_loss:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_reg_loss=False):

        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_reg_loss:
            return data_loss

        return data_loss, self.regularization_loss()

    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.weight_regulizer_l1 > 0:
                regularization_loss += layer.weight_regulizer_l1 * np.sum(np.abs(layer.weights))


            if layer.weight_regulizer_l2 > 0:
                regularization_loss += layer.weight_regulizer_l2 * np.sum(layer.weights ** 2)


            if layer.bias_regulizer_l1 > 0:
                regularization_loss += layer.bias_regulizer_l1 * np.sum(np.abs(layer.biases))


            if layer.bias_regulizer_l2 > 0:
                regularization_loss += layer.bias_regulizer_l2 * np.sum(layer.biases ** 2)

        return regularization_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, predictions, real):
        clipped_predictions = np.clip(predictions, 1e-7, 1-1e-7)

        if len(real.shape) == 1:
            confidences = clipped_predictions[range(len(clipped_predictions)), real]
        else:
            confidences = np.sum(clipped_predictions * real, axis=1)

        losses = -np.log(confidences)

        return losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        

class Loss_BinomialCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped)) - (1 - y_true) * np.log(1 - y_pred_clipped)
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Accuracy:

    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Accuracy_Categorical(Accuracy):

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2 and y.shape != predictions.shape:
            y = np.argmax(y, axis=1)

        return predictions == y


class SoftmaxActivation_CategoricalCrossEntropy(Loss):

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SDG:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:

            if not hasattr(layer, "weight_momentum"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights

            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:

    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * layer.dbiases  / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:

    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7, rho=.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache += self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases  / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:

    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7, beta1=.9, beta2=.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        corrected_weight_momentums = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        corrected_bias_momentums = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases ** 2

        corrected_weight_cache = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        corrected_bias_cache = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * corrected_weight_momentums / (np.sqrt(corrected_weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * corrected_bias_momentums  / (np.sqrt(corrected_bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1