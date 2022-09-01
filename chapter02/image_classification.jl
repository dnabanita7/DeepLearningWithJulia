using Flux, Zygote
using MLDatasets: CIFAR10
using MLUtils: splitobs
using Flux: onehotbatch
using Flux: DataLoader
using Flux: logitcrossentropy
using Flux: onecold

#Loading up our CIFAR10 training and testing datasets
images, labels = CIFAR10.traindata() # loading the train dataset
(train_images, train_labels), (val_images, val_labels) = splitobs((images, labels), at=0.9) #splitting the training dataset into 10% validation data
test_images, test_labels = CIFAR10.testdata() #loading the test dataset
#Data-preprocessing on our datasets
train_images = float(train_images) #floating pt. numbers are easy to handle
train_labels = onehotbatch(train_labels, 0:9) #coverting decimal to binary
val_images = float(val_images)
val_labels = onehotbatch(val_labels, 0:9)
test_images = float(test_images)
test_labels = onehotbatch(test_labels, 0:9)
#Loading and segregating our datasets
train_loader = DataLoader((train_images, train_labels), batchsize=128, shuffle=true) #loads data in an organized fashion
val_loader = DataLoader((val_images, val_labels), batchsize=128)
test_loader = DataLoader((test_images, test_labels), batchsize=128)
#Defining our model
model = Chain(
  Conv((5,5), 3=>16, relu),
  MaxPool((2,2)),
  Conv((5,5), 16=>8, relu),
  MaxPool((2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(200, 120),
  Dense(120, 84),
  Dense(84, 10),
  softmax) #multi-layered neural network model
#Defining loss function
loss(x, y) = logitcrossentropy(model(x), y)
#Defining optimizer function
opt = ADAM(3e-4)
ps = Flux.params(model)
#Training our model
for epoch in 1:50 # train on dataset for 50 times
    @info "Epoch $epoch"
    
    for (images, labels) in train_loader #getting 128 sized batches
        gs = Flux.gradient(() -> loss(images, labels), ps)
        Flux.update!(opt, ps, gs) #weight array is updated
    end
    
    validation_loss = 0f0 #floating point number
    for (images, labels) in val_loader #finding total validation loss
        validation_loss += loss(images, labels)
    end
    validation_loss /= length(val_loader) #finding mean validation loss
    @show validation_loss
end
#Check our model's performance on test dataset
correct, total = 0, 0
for (images, labels) in test_loader # Test model on unseen data
correct += sum(onecold(model(images)) .== onecold(labels))
total += size(labels, 2)
end
test_accuracy = correct / total
@show test_accuracy
