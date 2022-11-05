using CSV, DataFrames # importing the libraries for working with CSV files with dataframe objects
using MLBase, MLDataUtils, Random

df = CSV.File("seeds_dataset.txt") |> DataFrame # Converting a text file dataset into a CSV file by using a csv file reader and then piping the dataset into a dataframe for easy data manipulation
columns = names(df) # get the column names

for name in columns # normalizing all the values in the dataset for easily fitting in the data
    for val in df[!, name] # get the values in specific columns
        val = (val - minimum([df[!, name]]...)) / (maximum([df[!, name]]...) - minimum([df[!, name]]...))
    end
end

df = df[shuffle(1:end), :] # shuffling the dataset before splitting them into train/test datasets
folds = kfolds(df, 5) # cross-validate the dataset into equal percentage of categorical values in #each split

function initialize(numinput, numhidden, numoutput) # create a model from scratch with a single #hidden layer
     model = []
     inputLayer = []
     hiddenLayer = []
     hiddenLayer = [Dict("weights" => [rand() for i in range(1, numinput+1)]) for i in range(1, numhidden)] # adding a hidden layer to the model
     push!(model, hiddenLayer)
     outputLayer = []
     hiddenLayer = []
     outputLayer = [Dict("weights" => [rand() for i in range(1, numhidden+1)]) for i in range(1, numoutput)] # adding the output layer to the model
     push!(model, outputLayer)
     outputLayer = []
     return model
end

function activate(weights, inputs) # activating the weights of the neurons in the layers
    activation = last(weights)
    for i in range(1, length(weights)-1)
        activation = activation + weights[i] * inputs[i]
    end
    return activation
end

function transfer(activation) # activating neurons using sigmoid functions
    return 1.0 / (1.0 + exp(-1 * activation))
end

function forwardPropagate(network, row) # pass in weights and evaluate the model from input input to output layer through hidden layer
    inputs = row
    newInputs = []
    for layer in network
        newIn = []
        activation = 0.0
        for neuron in layer
            activation = activate(neuron["weights"], inputs)
        end
        push!(layer, Dict("output" => [transfer(activation)]))
        push!(newIn, transfer(activation))
        newInputs = newIn
    end
    inputs = newInputs
    return inputs
end

function derivative(output) # find derivatives of each of the neurons
    return output * (1.0 - output)
end


function backPropagate(network, expected) # back propagate the error through the network
           for i in length(network):1:-1
               layer = network[i]
               errors = []
               if i != length(network)-1
                   for j in 1:length(layer)
                       error = 0.0
                       for neuron in network[i+1]
                           error = error + neuron["weights"][j] * neuron["delta"]
                       end
                       push!(errors, error)
                   end
               else
                   for j in 1:length(layer)
                       neuron = layer[j]
                       push!(errors, neuron["output"] - expected[j])
                   end
               end
    delta = []
           for j in 1:length(layer)
               neuron = layer[j]
               push!(delta, errors[j] * derivative(neuron["output"]))
    push!(layer, Dict(“delta”=>delta))
    delta = []
end
   end
end


function updateWeights(network, row, lrate) # Update network weights with respect to the error
    for i in 1:length(network)
        inputs = row[:length(row)]
        if i != 0
            inputs = [neuron['output'] for neuron in network[i - 1]]
	end
        for neuron in network[i]
            for j in 1:length(inputs)
                neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
	    end
            neuron['weights'][length(neuron[‘weights’])] -= lrate * neuron['delta']
        end
     end
end
function train(network, train, lrate, numepoch, numoutput) # Train a network for a fixed number of epochs
    for epoch in 1:numepoch
        sumErr = 0
        for row in train
            outputs = forwardPropagate(network, row)
            expected = [0 for i in 1:numoutput]
            expected[row[length(row)]] = 1
            sumErr = sumErr + sum([(expected[i]-outputs[i])^2 for i in 1:length(expected)])
            backPropagate(network, expected)
            updateWeights(network, row, lrate)
	end
     end
        print(“Epoch: ”, epoch)
	print(“Loss Rate: ”, lrate)
	print(“Sum of all errors: ”, sum_error)
end

function predict(network, row) # Make a prediction with a network
    outputs = forwardPropagate(network, row)
    return outputs.index(max(outputs))
end

    
function accuracy(actual, predicted) # Calculate accuracy percentage
        correct = 0
        for i in 1:length(actual)
            if actual[i] == predicted[i]
                correct += 1
	    end
	end
        return correct / float(length(actual)) * 100.0
end
 

function evaluate(dataset, algorithm, *args) # Evaluate an algorithm using a cross validation split
        scores = []
        for fold in folds
            trainSet = [folds]
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = []
        end
        for row in fold
            row1 = [row]
            push!(testSet, row1)
            row1[length(row)] = nothing
        end
        predicted = algorithm(trainSet, testSet, *args)
        actual = [row[length(row)] for row in fold]
        accuracy = accuracy(actual, predicted)
        push!(scores, accuracy)
        return scores
end

lrate = 0.3
numepoch = 500
numhidden = 5
row = [1, 0, 1]
scores = evaluate(dataset, backPropagate, lrate, numepoch, numhidden)
print(“Scores: “, scores)
print(“Mean Accuracy: “, (sum(scores)/float(length(scores))))
