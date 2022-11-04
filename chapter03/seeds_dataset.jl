using CSV, DataFrames

df = CSV.File("seeds_dataset.txt") |> DataFrame
210×8 DataFrame
 Row │ Area     Perimeter  Compactness  Kernel Length  Kernel Width  Asymmetry ⋯
     │ Float64  Float64    Float64      Float64        Float64       Float64   ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │   15.26      14.84       0.871           5.763         3.312            ⋯
   2 │   14.88      14.57       0.8811          5.554         3.333
   3 │   14.29      14.09       0.905           5.291         3.337
   4 │   13.84      13.94       0.8955          5.324         3.379
   5 │   16.14      14.99       0.9034          5.658         3.562            ⋯
   6 │   14.38      14.21       0.8951          5.386         3.312
   7 │   14.69      14.49       0.8799          5.563         3.259
   8 │   14.11      14.1        0.8911          5.42          3.302
  ⋮  │    ⋮         ⋮           ⋮             ⋮             ⋮                  ⋱
 204 │   12.7       13.41       0.8874          5.183         3.091            ⋯

columns = names(df)
8-element Vector{String}:
 "Area"
 "Perimeter"
 "Compactness"
 "Kernel Length"
 "Kernel Width"
 "Asymmetry Coefficient"
 "Kernel Groove"
 "Type"


for name in columns
           for val in df[!, name]
                  val = (val - minimum([df[!, name]]...)) / (maximum([df[!, name]]...) - minimum([df[!, name]]...))
                  print(val)
              end
           end
0.4409820585457979 0.40509915014164316 0.3493862134088762 …

using MLBase, MLDataUtils

using Random

df = df[shuffle(1:end), :]
210×8 DataFrame
 Row │ Area     Perimeter  Compactness  Kernel Length  Kernel Width  Asymmetry Co ⋯
     │ Float64  Float64    Float64      Float64        Float64       Float64      ⋯
─────┼─────────────────────────────────────────────────────────────────────────────
   1 │   17.63      15.86       0.88            6.033         3.573               ⋯
   2 │   11.41      12.95       0.856           5.09          2.775
   3 │   12.13      13.73       0.8081          5.394         2.745
   4 │   17.99      15.86       0.8992          5.89          3.694
   5 │   11.27      12.97       0.8419          5.088         2.763               ⋯

folds = kfolds(df, 5) # cross-validate
5-fold FoldsView of 210 observations:
  data: 210×8 DataFrame
  training: 168 observations/fold
  validation: 42 observations/fold
  obsdim: "NA"

function initialize(numinput, numhidden, numoutput)
           model = []
           inputLayer = []
           hiddenLayer = []
           hiddenLayer = [Dict("weights" => [rand() for i in range(1, numinput+1)]) for i in range(1, numhidden)]
           push!(model, hiddenLayer)
           outputLayer = []
           hiddenLayer = []
           outputLayer = [Dict("weights" => [rand() for i in range(1, numhidden+1)]) for i in range(1, numoutput)]
           push!(model, outputLayer)
           outputLayer = []
           return model
       end
initialize (generic function with 1 method)

function activate(weights, inputs)
           activation = last(weights)
           for i in range(1, length(weights)-1)
               activation = activation + weights[i] * inputs[i]
           end
           return activation
       End
function transfer(activation)
           return 1.0 / (1.0 + exp(-1 * activation))
       End

function forwardPropagate(network, row)
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
forwardPropagate (generic function with 1 method)


function derivative(output)
           return output * (1.0 - output)
       end


function backPropagate(network, expected)
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


function updateWeights(network, row, lrate) # Update network weights with error
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
scores = evaluate(dataset, backPropagate, lrate, numepoch, numhidden)
print(“Scores: “, scores)
print(“Mean Accuracy: “, (sum(scores)/float(length(scores))))
