using Pkg
Pkg.activate(".")
Pkg.add(["Flux", "RDatasets", "MLUtils", "Statistics", "Plots", "Random"])

using Flux
using RDatasets
using MLUtils
using Statistics
using Plots
using Random

# -------------------------
# 1. Load and Prepare Data
# -------------------------
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])'  # size: (4, 150)
labels = iris[:, :Species]

# Encode labels
label_map = Dict("setosa" => 1, "versicolor" => 2, "virginica" => 3)
y_int = [label_map[string(lbl)] for lbl in labels]
Y = Flux.onehotbatch(y_int, 1:3)

# Normalize features
X = (X .- mean(X, dims=2)) ./ std(X, dims=2)

# Split and shuffle
Random.seed!(69)
datasett = [(X[:, i], Y[:, i]) for i in 1:size(X, 2)]
train_data, test_data = splitobs(shuffleobs(datasett), at=0.8)

# -------------------------
# 2. Define Model
# -------------------------
model = Chain(
    Dense(4, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 3),
    softmax
)

loss(x, y) = Flux.crossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))

opt_state = Flux.setup(Adam(), model)

# -------------------------
# 3. Training Loop
# -------------------------
epochs = 50
train_loss = Float64[]
train_acc = Float64[]

for epoch in 1:epochs
    for (x, y) in train_data
        x = Float32.(x)  # Ensure correct type
        y = Float32.(y)
        gs = Flux.gradient(model) do m
            loss(x, y)
        end
        Flux.update!(opt_state, model, gs[1])
    end
    l = loss(hcat(first.(train_data)...), hcat(last.(train_data)...))
    a = accuracy(hcat(first.(train_data)...), hcat(last.(train_data)...))
    push!(train_loss, l)
    push!(train_acc, a)
    println("Epoch $epoch - Loss: $(round(l, digits=4)) | Accuracy: $(round(a * 100, digits=2))%")
end

# -------------------------
# 4. Evaluate
# -------------------------
test_X = hcat(first.(test_data)...)
test_Y = hcat(last.(test_data)...)
println("Test Accuracy: ", round(accuracy(test_X, test_Y) * 100, digits=2), "%")

# -------------------------
# 5. Plots
# -------------------------
plot(train_loss, xlabel="Epoch", ylabel="Loss", title="Training Loss", legend=false)
plot(train_acc, xlabel="Epoch", ylabel="Accuracy", title="Training Accuracy", legend=false)
