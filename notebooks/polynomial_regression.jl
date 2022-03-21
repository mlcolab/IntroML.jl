### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(
                Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                "AbstractPlutoDingetjes",
            )].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ef43aef6-80ec-4976-91e6-84a74d29a83e
begin
    using PlutoUI,
        Random, StatsPlots, LinearAlgebra, Turing, LaTeXStrings, DataFrames, Optim
    using Turing: Flat
end

# ╔═╡ 2c05ac8e-7e0b-4d28-ad45-24e9c21aa882
md"""
# Introduction

In this example, we introduce some basic concepts of machine learning using regression as an example.
"""

# ╔═╡ 18198312-54c3-4b4f-b865-3a6a775ce483
md"""
# The data

At regularly spaced points ``x``, we have generated fake, noisy observations ``y``.
"""

# ╔═╡ cd49e0a5-4120-481a-965e-72e7bdaf867c
md"""
# Regression models

We want to construct a model that makes a prediction ``\hat{y}`` given ``x``.
One way to do this is with regression.
The simplest possible model we could fit is a horizontal line:
````math
\hat{y} = w_0
````
"""

# ╔═╡ 8eb34f0c-166b-4973-bdea-2ae64e3d34aa
w0_slider = @bind w0 Slider(-5:0.001:5; default=0.3, show_value=true);

# ╔═╡ 61238019-df77-4067-a9b4-0766d056e891
md"``w_0 = `` $w0_slider"

# ╔═╡ 5e7bda42-0266-4498-906d-9aca8b6c4bf3
f(x) = sinpi(2x) + 2.5x

# ╔═╡ 3f4bffbb-51f4-4446-9b9c-cd3f99edfefa
lossl2(yhat, y) = sum(abs2, yhat .- y) / 2

# ╔═╡ 04c7209e-c4f8-454b-a883-cb2c5fac5203
error(f_hat, x, y) = lossl2(f_hat.(x), y)

# ╔═╡ 0b719897-4515-46c3-830a-eaec4d1666a2
md"``w_0 = `` $w0_slider"

# ╔═╡ 670bb91c-5b84-4e93-8288-90734f92b4f2
w1_slider = @bind w1 Slider(-10:0.01:10; default=-3, show_value=true);

# ╔═╡ 1159b9d5-d1b9-4f6d-b181-1f4e6fa9c0cc
md"""
``w_0 =`` $w0_slider ``\quad`` ``w_1 =`` $w1_slider
"""

# ╔═╡ 7f9e91b8-ee23-4d73-bfe5-c58a29b77abe
md"""
# Automatically fitting parameters

There are two problems with how we have been fitting parameters so far.
First, it's manual.
This won't scale to more than a few parameters.
Second, it's fitting 1 parameter at a time, when ideally we would fit all at the same time.
How can we automate the computer doing this for us?
"""

# ╔═╡ 876fa74c-9c30-4f0b-9a5b-82bb6597cd47
g = [zero]

# ╔═╡ 06e8320c-ddd9-4d13-bca3-10fb5c3fb7ad
md"""
This approach of manually selecting features works well when:
1. we already can guess quite a lot about the function without looking at the data
2. The function is simple enough that a very small number of features can be linearly combined to approximate it
3. the data is small and simple enough that we can plot it and guess what features might be useful.

Often in machine learning applications, we're not so lucky, so we need generic approaches to construct useful features.
"""

# ╔═╡ 1f1e9c9b-e5fa-41b1-852f-cadad703ee4b
md"""
# Model complexity

We see that for low ``n``, we don't fit the data very well, but by ``n=3`` we get quite a good fit.
By the time we reach ``n=9``, the model is so flexible, it's able to perfectly hit every data point, so that the loss is 0.
As we increase ``n``, additional weights take on very large values, even though they cannot fit the data any better.

Question: which model is best?
"""

# ╔═╡ 7fa55e99-5c0c-465d-a879-bd844e516131
error_rms(fhat, x, y; method=error) = sqrt(method(fhat, x, y) * 2//length(x))

# ╔═╡ 6fb68c61-1ef0-4efc-bcbc-dd9d219c3ebb
md"""
# Regularization

It may seem a little strange that adding more weights can make the model worse, since the simpler model is contained within the more compplex one.
One way to understand this is to look at the fitted values of the weights as we add more terms.
"""

# ╔═╡ c50f50f4-84a3-4a81-bacc-b8ce99d6b257
md"""
Watch what happens as we increase the number of data points.
We can increase the number of degrees of freedom of the model without getting the wild oscillations that we saw when the data was more sparse.
However, this does not mean that our error on the test data continues to decrease.
For 1000 data points, we find that the minimum ``E_\mathrm{RMS}`` is at ``n=5``.
"""

# ╔═╡ a2f40cbe-f46f-44c6-a81f-aae082c27c1a
logλ_input = @bind logλ Slider([-Inf; -20:1:0]; default=-Inf, show_value=true)

# ╔═╡ 7e532caa-b8a5-43ab-af4a-4590ffd1a9dc
@model function poly_regress_bayes(
    x,
    y,
    max_order;
    σ=0.2,
    λ=0,
    σw=sqrt(inv(λ)) * σ, # convert regularization to standard deviation
    wprior=iszero(λ) ? Flat() : Normal(0, σw),
    p=0:max_order,
    features=x .^ p',
)
    w ~ filldist(wprior, max_order + 1)
    return y .~ Normal.(features * w, σ)
end

# ╔═╡ 3fc67210-6df1-481b-8aa4-48865f9f46b5
md"""
What we have been doing so far is computing the maximum-likelihood (a posteriori) estimate of this regression model, i.e. the set of parameters that distribution of parameters given data has its peak [show visualization].
To quantify the uncertainty, we can instead visualize the distribution of parameters.
Each set of parameters defines a curve, so we can visualize this distribution by visualizing random curves:
"""

# ╔═╡ 3991be1c-01f3-416a-a841-18f025a97e24
md"""
# Regression for classification problems

Now, ``f`` maps from continus features to continuous observations, but what about a binary classification problem, where ``y`` can take on values only ``0`` and ``1``?
[show data]
"""

# ╔═╡ ed7cb5c1-528a-4356-80dd-b337107eaf1f
md"""
[show fit]
"""

# ╔═╡ 5b237453-472f-414e-95e0-f44e980ea93a
md"""
# Utilities

This section contains data generation, utility functions and UI elements used in the above notebook.
"""

# ╔═╡ f75ad936-8c06-4e00-92d7-1f86532c0072
TableOfContents()

# ╔═╡ c75744a0-3c3f-4042-a796-6cbd9ec11195
md"""
## UI elements

This section contains UI elements and variables they are bound to.
"""

# ╔═╡ 2cc52188-b262-4f65-b042-ad94d90523d8
npoints_input = @bind npoints NumberField(1:1_000; default=10);

# ╔═╡ f32db22e-d111-4bf5-9989-a698b0d22626
npoints_input

# ╔═╡ 4b98bd17-de33-4648-b737-6b175905b2c7
max_order_input = @bind max_order NumberField(0:100; default=0);

# ╔═╡ b0cdc9d6-738a-4583-b821-052ada846d39
max_order_input

# ╔═╡ 06dee467-1f54-48e1-908d-8e4c9028a748
max_order_input

# ╔═╡ efb34c1a-5505-49f1-aa7f-24f6fd1fc01d
max_order_input

# ╔═╡ e2890775-2e29-4244-adac-c37f8f2a8a8e
max_order_input

# ╔═╡ b176823e-b8b5-413d-87b1-90d7efa0e377
important(text) = HTML("""<span style="color:magenta"><strong>$text</strong></span>""")

# ╔═╡ 4c7e53c5-1271-4acf-95cc-5345564d1b15
md"""
But which is the best fit?
The one that minimizes the error.
So we need a notion of error, and then we want to find the line with the smallest error.
This is called a $(important("loss function")).
Here, we choose the ``\ell^2`` loss function:
````math
L(\hat{y}, y) = \frac{1}{2}\sum_{i=1}^n (\hat{y}_i - y_i)^2
````
"""

# ╔═╡ ae5d8669-f4c4-4b55-9af9-8488e43bcb6c
md"""
# Linear regression

For a better fit, let's give the line not only an intercept ``w_0`` but also also a slope ``w_1``:

````math
\hat{y} = w_0 + w_1 x
````

``w = \begin{pmatrix}w_0 \\ w_1\end{pmatrix}`` is called a $(important("weight")) vector.
"""

# ╔═╡ 6f75889b-1c7f-4261-bf27-7c991ee9e414
md"""
Here we're plotting a contour map for the value of the loss function for each pair of weight values.
Take a look at the trajectory the computer followed.
At each point, the next step is in a direction perpendicular to the level curve at that point.
This is the direction of steepest descent.
For any set of weights, a modern machine learning package can automatically compute this direction using a method called $(important("backpropagation")), where it approximates how much each parameter is responsible for the error and should therefore change to minimize the error.
An $(important("optimizer")) uses the direction of steepest descent to pick the next weights to try.

In practice, the packages do this for you automatically so that you never need to backpropagate yourself.
However, there are many more sophisticated optimizers than this one that rely on backpropagation.

From now on, we will ignore backpropagation and the optimizer and treat them as a black box.
"""

# ╔═╡ 74290eff-781b-44c9-8a90-96bffbe040df
md"""
# Model expansion

Fitting the model gave us the best linear fit, which is obviously not great.
To get a better fit, we need to choose a different form for ``f(x)``.
Alternatively, we can keep ``f(x)`` and train the model on $(important("features")) computed from ``x`` instead of (or in addition to) ``x`` itself.<span style="color:blue"> blah </span>

Let ``g_j`` be a function that computes some feature ``j`` from ``x``.
We'll approximate ``f`` as a linear combination of these features:

```math
\hat{y} = \sum_{j=1}^n w_j g_j(x)
```

Note that if ``g_j`` is a nonlinear function of ``x``, then ``\hat{y}`` is a nonlinear function of ``x`` but still a linear function of the weight vector ``w``.

Let's try guessing useful scalar functions to add to ``g`` below:
"""

# ╔═╡ ca1f0910-d417-41bc-ae2d-eebec7f3e1e9
md"""
# Polynomial regression

One approach is to select increasing powers of ``x`` as the features.

```math
\hat{y}_i = w_0 + w_1 x_1 + w_2 x^2 \ldots w_n x^n = \sum_{j=0}^n w_j x_i^j
```

Functions of the form of ``f`` are called $(important("polynomials")), and any smooth function can be exactly computed with infinite terms (i.e. ``n \to \infty``) or approximated with finite terms (by picking some manageable ``n``).

Linear regression is the special case $n=1$, though note again that here ``\hat{y}`` is still linear with respect to the weights ``w``.
"""

# ╔═╡ 39414e5e-1256-4497-9738-e2ecdff62d9d
md"""
# Cross-validation and model selection

The goal is to have a model that not only fits the data well but has also learned something about the target function.
That is, we want a model that also generalizes well to data that was not used to fit it.
To check this for our models, we use a test dataset ``(x_\mathrm{test}, y_\mathrm{test})``, data similar to ``(x, y)`` that we only use to test our approximate function.

To evaluate error, we use the scalaed $(important("root-mean-squared error")):
```math
E_\mathrm{RMS} = \sqrt{\frac{2L(\hat{y}, y)}{n}}
```
"""

# ╔═╡ fe2c7c2b-63e3-4cbf-b432-b028ec599292
md"""
Ideally, we would have an approach that avoids overfitting regardless of the amount of data being used.
We can in fact do this by $(important("regularizing")) the weights.
In effect, we add a term to our loss function that penalizes large weight values.
Our regularized error function is

```math
\tilde{E}(w) = E(w) + \frac{\lambda}{2}\sum_{i=1}^n w_i^2.
```

The strength of the penalization is controlled by the magnitude of ``\lambda``.
When ``\lambda=0``, we apply no regularization, and the weights are fit only by the data.
When ``\lambda`` is very large, the regularization term dominates, and it takes very strong signal in the data to move the weights away from 0.
"""

# ╔═╡ 73444608-d0de-440d-8be3-5a02dadcadc7
md"""
# Handling uncertainty

All along we've been showing the single best fitting function, but data is noisy and incomplete, so we can't be 100% certain about what that best fitting function is.
Regression models are statistical models, and specifically, we can write them as Bayesian models.
In a $(important("Bayesian")) model (AKA a "probabilistic model"), uncertainties are quantified with probabilities.

Bayesian models combine a likelihood function (i.e. our loss function) with a prior on the model parameters (i.e. our regularization term), which defines distribution of parameters (i.e. our weights) given the data.
"""

# ╔═╡ ba59678b-1606-4132-9f19-0dae1e660195
md"""
We can keep almost everything the same here, but what we need to do is add a $(important("sigmoid"))al (S-shaped) nonlinearity to map from the real output of ``f`` to the interval ``[0, 1]``.
We can choose the logistic function as our sigmoid, and the result is logistic regression:
[formula]
"""

# ╔═╡ d8983a9d-1880-4dc4-9c17-23281767e0c2
md"## Definitions"

# ╔═╡ ddee1cf7-7977-407d-a004-08b52f6ff8c8
md"## Plotting functions"

# ╔═╡ 485c046d-8329-4541-bd9d-eb180c01bde6
function plot_data!(
    p,
    x,
    y;
    xrange=(0, 1),
    f_actual=nothing,
    f_hat=nothing,
    f_draws=nothing,
    show_residuals=false,
    data_color=:blue,
)
    if show_residuals && f_hat !== nothing
        sticks!(p, x, y; fillrange=f_hat.(x), color=:red)
    end
    scatter!(p, x, y; xlabel=L"x", ylabel=L"y", color=data_color, msw=0, ms=3)
    if f_actual !== nothing
        plot!(p, f_actual, xrange...; color=:green, lw=2)
    end
    if f_draws !== nothing
        for fi in f_draws
            plot!(p, fi, xrange...; color=:orange, lw=1, label="", alpha=0.2)
        end
    end
    if f_hat !== nothing
        plot!(p, f_hat, xrange...; color=:orange, lw=2, label="")
    end
    plot!(p; legend=false)
    return p
end

# ╔═╡ b9318fcf-117e-438e-8bb4-985a9372e2d8
plot_data(x, y; kwargs...) = plot_data!(plot(), x, y; kwargs...)

# ╔═╡ ba0545e7-c6df-42e2-a9cd-4ecd490d13e8
md"""
## Data generation
"""

# ╔═╡ 449f1aae-57b4-4b24-98d9-32c514e00821
error_actual = 0.2

# ╔═╡ 510f07d9-62e0-40a7-b974-e2ae9bad7f73
function generate_data(f, x, error; rng=Random.GLOBAL_RNG)
    return f.(x) .+ randn.(rng) .* error
end

# ╔═╡ 905793d5-93c5-4d86-9a88-33d6d806d88a
ndata = 10

# ╔═╡ c17da7e0-d701-45e6-9967-b2cb0d2b057e
data_seed = 42

# ╔═╡ d9a82c82-576e-4ee5-a8be-aea1332b3e74
data_test_seed = 63

# ╔═╡ b77903d1-c788-4efd-80c4-313859e856e5
x = collect(range(0, 1; length=ndata))

# ╔═╡ 01af74cf-172d-4561-a7d9-6131a22b4161
y = generate_data(f, x, error_actual; rng=MersenneTwister(data_seed))

# ╔═╡ 1b4812d4-3879-4a79-a95d-20cad2959f5c
plot_data(x, y)

# ╔═╡ 72198269-d070-493f-92eb-36135692ca8f
plot_data(x, y; f_hat=x -> w0, show_residuals=true)

# ╔═╡ 288aa7d7-8785-4f55-95e6-409e2ceb203a
let
    f_hat(x) = w0
    loss = round(error(f_hat, x, y); digits=2)
    p = plot_data(x, y; f_hat, show_residuals=true)
    plot!(p; title="loss: $loss")
end

# ╔═╡ 345ae96b-92c2-4ac4-bfdf-302113627ffb
let
    f_hat(x) = w0 + w1 * x
    loss = round(error(f_hat, x, y); digits=2)
    p = plot_data(x, y; f_hat, show_residuals=true)
    plot!(p; title="loss: $loss")
end

# ╔═╡ 0d1164df-8236-494b-b8b9-71481c94c0d9
let
    f_hat(x) = w0 + w1 * x
    loss = round(error(f_hat, x, y); digits=2)
    scatter([w0], [w1]; xlims=(-5, 5), ylims=(-10, 10))
end

# ╔═╡ 942f314e-927b-4371-8c83-83801c860b4d
line_trace = let
    obj(w) = error(x -> w[1] + w[2] * x, x, y)
    winit = [0.0, 0.0]
    optimizer = Optim.GradientDescent(; linesearch=Optim.LineSearches.BackTracking())
    options = Optim.Options(; store_trace=true, extended_trace=true)
    res = Optim.optimize(obj, winit, optimizer, options)
    wtrace = Optim.x_trace(res)
end

# ╔═╡ 6016a736-11da-4451-aa82-cc3045e782db
let
    obj(w) = error(x -> w[1] + w[2] * x, x, y)
    plot(first.(line_trace), last.(line_trace); seriestype=:scatterpath, ms=2, msw=0)
    scatter!([w0], [w1]; xlabel="w0", ylabel="w1", msw=0)
    contour!(-1:0.01:2, -1:0.01:2, (w0, w1) -> obj([w0, w1]); levels=100)
    plot!(; aspect_ratio=1)
end

# ╔═╡ 2e683999-0d94-47a1-ab62-cd10b8c3b300
chns = let
    mod = poly_regress_bayes(x, y, 10)
    sample(mod, NUTS(), 500)
end

# ╔═╡ 5def5552-fc0e-4e44-bed2-49edd810c75a
ytest = generate_data(f, x, error_actual; rng=MersenneTwister(data_test_seed))

# ╔═╡ a2e8fdca-7f23-4f94-8b94-502ff29500bc
xmore = collect(range(0, 1; length=npoints))

# ╔═╡ 072bf511-749d-421b-8e56-d706da88f031
ymore = generate_data(f, xmore, error_actual; rng=MersenneTwister(data_seed))

# ╔═╡ f6f3c9b7-dd9d-4f7b-9626-93534c15f199
ymore_test = generate_data(f, xmore, error_actual; rng=MersenneTwister(data_test_seed))

# ╔═╡ 318a6643-5377-4152-8468-51dae1b78144
md"""
## Model fitting
"""

# ╔═╡ bddafce9-30e4-4708-96ae-938bff9edfe7
solve_regression(x, y) = x \ y

# ╔═╡ ebb0c754-12f1-4f80-a5f6-98a61b915fa6
solve_regression(x, y, λ) = (x'x + λ * I) \ (x'y) # ridge regression

# ╔═╡ a5fe54fb-4f92-4a35-8038-8d36a4aa065c
begin
    struct PolyModel{T<:Real,L<:Real,W<:AbstractVector{T}}
        w::W
        λ::L
    end
    PolyModel(max_order::Int, λ=0) = PolyModel(zeros(max_order + 1), λ)
    PolyModel(w) = PolyModel(w, 0)

    function (m::PolyModel)(x)
        w = m.w
        p = 0:(length(w) - 1)
        return dot(x .^ p, w)
    end

    function fit!(m::PolyModel, x, y)
        w = m.w
        p = 0:(length(w) - 1)
        features = x .^ p'
        if iszero(m.λ)
            w .= solve_regression(features, y)
        else
            w .= solve_regression(features, y, m.λ)
        end
        return m
    end
end;

# ╔═╡ 3b7fe147-ea94-422f-a943-6f3bd577edf1
f_hat_poly = fit!(PolyModel(max_order), x, y)

# ╔═╡ fc64996f-54ba-4c7a-8cfb-21133cec1fbe
let
    err = round(error(f_hat_poly, x, y); digits=2)
    p = plot_data(x, y; f_hat=x -> f_hat_poly(x), show_residuals=true)
    plot!(p; title="\$E = $err \$")
end

# ╔═╡ e3dbe2f5-7e97-45b7-9b75-1acaf8f1031b
let
    plots = map([0, 1, 3, 9]) do max_order
        f_hat = fit!(PolyModel(max_order), x, y)
        p = plot_data(x, y; f_hat=x -> f_hat(x), show_residuals=false)
        plot!(p; title="\$n=$max_order\$")
    end
    plot(plots...)
end

# ╔═╡ aa7b8b58-f959-47de-84d7-8c9cf3ad96be
let
    p = plot_data(
        x, ytest; f_hat=x -> f_hat_poly(x), data_color=:magenta, show_residuals=true
    )
    plot_data!(p, x, y)
    max_orders = 0:max(10, max_order)
    rmse_values = map(max_orders) do n
        m = fit!(PolyModel(n), x, y)
        return error_rms(m, x, y), error_rms(m, x, ytest)
    end
    p2 = plot(max_orders, first.(rmse_values); color=:blue, lw=2, label="training")
    plot!(p2, max_orders, last.(rmse_values); color=:magenta, lw=2, label="test")
    scatter!(
        p2,
        [max_order],
        [rmse_values[max_order + 1]...]';
        color=[:blue :magenta],
        msw=0,
        ms=3,
        label="",
    )
    plot!(p2; legend=:topright, xlabel=L"n", ylabel=L"E_\mathrm{RMS}", ylims=(-0.01, NaN))
    plot(p, p2)
end

# ╔═╡ a9163072-cad9-4b0b-b154-d315c6b68de4
let
    max_orders = [0, 1, 3, 6, 9]
    pairs = map(max_orders) do n
        m = fit!(PolyModel(n), x, y)
        # pad with `missing`s
        "$n" => [m.w; fill(md"", maximum(max_orders) - n)]
    end
    DataFrame(pairs)
end

# ╔═╡ 444e4eba-9b5a-4e37-8853-5d24c5c398ca
let
    x, y, ytest = xmore, ymore, ymore_test
    f_hat = fit!(PolyModel(max_order), x, y)
    p = plot_data(x, ytest; f_hat=x -> f_hat(x), data_color=:magenta, show_residuals=false)
    # plot_data!(p, x, y)
    max_orders = 0:max(10, max_order)
    rmse_values = map(max_orders) do n
        m = fit!(PolyModel(n), x, y)
        return error_rms(m, x, y), error_rms(m, x, ytest)
    end
    p2 = plot(max_orders, first.(rmse_values); color=:blue, lw=2, label="training")
    plot!(p2, max_orders, last.(rmse_values); color=:magenta, lw=2, label="test")
    scatter!(
        p2,
        [max_order],
        [rmse_values[max_order + 1]...]';
        color=[:blue :magenta],
        msw=0,
        ms=3,
        label="",
    )
    plot!(p2; legend=:topright, xlabel=L"n", ylabel=L"E_\mathrm{RMS}", ylims=(-0.01, NaN))
    plot(p, p2)
end

# ╔═╡ 63d99d6f-addf-4efc-a680-d4c4733e3941
error_reg(m::PolyModel, x, y) = error(m, x, y) + sum(abs2, m.w) * m.λ / 2

# ╔═╡ 645525f1-77cb-4b18-81df-3eafc0b4004e
let
    λ = exp(logλ)
    f_hat_poly = fit!(PolyModel(max_order, λ), x, y)
    p = plot_data(
        x, ytest; f_hat=x -> f_hat_poly(x), data_color=:magenta, show_residuals=true
    )
    plot_data!(p, x, y)
    max_orders = 0:max(10, max_order)
    rmse_values = map(max_orders) do n
        m = fit!(PolyModel(n, λ), x, y)
        return error_rms(m, x, y), error_rms(m, x, ytest)
    end
    p2 = plot(max_orders, first.(rmse_values); color=:blue, lw=2, label="training")
    plot!(p2, max_orders, last.(rmse_values); color=:magenta, lw=2, label="test")
    scatter!(
        p2,
        [max_order],
        [rmse_values[max_order + 1]...]';
        color=[:blue :magenta],
        msw=0,
        ms=3,
        label="",
    )
    plot!(p2; legend=:topright, xlabel=L"n", ylabel=L"E_\mathrm{RMS}", ylims=(-0.01, NaN))
    plot(p, p2)
end

# ╔═╡ 35e70aa4-7b87-4730-8e05-f39d29c20a9e
function rand_weights(chns, n=10; rng=Random.GLOBAL_RNG)
    warray = permutedims(cat(get(chns, :w).w...; dims=3), (3, 2, 1))
    wdraws = reshape(warray, size(warray, 1), :)
    inds = rand(rng, axes(wdraws, 2), n)
    return [wdraws[:, i] for i in inds]
end

# ╔═╡ 8ea2e159-ef17-4ddd-b5a7-5f6c8d67238a
let
    plots = map([0, 1, 3, 9]) do max_order
        mod = poly_regress_bayes(x, y, max_order)
        chns = sample(mod, NUTS(), 500)
        f_draws = map(rand_weights(chns, 20)) do w
            f = PolyModel(w)
            return x -> f(x)
        end
        f_hat = PolyModel(optimize(mod, MAP()).values)
        p = plot_data(x, y; f_draws, f_hat=x -> f_hat(x))
        plot!(p; title=L"n=$max_order")
        p
    end
    plot(plots...)
end

# ╔═╡ 4748a526-8d2e-43a6-8f30-82abf238d624
begin
    # compute feature matrix
    function compute_features(g, x)
        z = similar(x, length(x)..., length(g)...)
        for j in eachindex(g)
            z[:, j] .= g[j].(x)
        end
        return z
    end
    # compute feature vector
    compute_features(g, x::Real) = map(gj -> gj(x), g)
end

# ╔═╡ 34bad558-e70f-4d46-a9ab-7acc6c89db7a
let
    w = solve_regression(compute_features(g, x), y)
    f_hat(x) = dot(compute_features(g, x), w)
    err = round(error(f_hat, x, y); digits=2)
    p = plot_data(x, y; f_hat, show_residuals=true)
    plot!(p; title="\$E = $err \$")
end

# ╔═╡ Cell order:
# ╟─2c05ac8e-7e0b-4d28-ad45-24e9c21aa882
# ╟─18198312-54c3-4b4f-b865-3a6a775ce483
# ╠═1b4812d4-3879-4a79-a95d-20cad2959f5c
# ╟─cd49e0a5-4120-481a-965e-72e7bdaf867c
# ╠═8eb34f0c-166b-4973-bdea-2ae64e3d34aa
# ╟─61238019-df77-4067-a9b4-0766d056e891
# ╠═72198269-d070-493f-92eb-36135692ca8f
# ╠═5e7bda42-0266-4498-906d-9aca8b6c4bf3
# ╟─4c7e53c5-1271-4acf-95cc-5345564d1b15
# ╠═3f4bffbb-51f4-4446-9b9c-cd3f99edfefa
# ╠═04c7209e-c4f8-454b-a883-cb2c5fac5203
# ╟─0b719897-4515-46c3-830a-eaec4d1666a2
# ╠═288aa7d7-8785-4f55-95e6-409e2ceb203a
# ╟─ae5d8669-f4c4-4b55-9af9-8488e43bcb6c
# ╠═670bb91c-5b84-4e93-8288-90734f92b4f2
# ╟─1159b9d5-d1b9-4f6d-b181-1f4e6fa9c0cc
# ╠═345ae96b-92c2-4ac4-bfdf-302113627ffb
# ╟─7f9e91b8-ee23-4d73-bfe5-c58a29b77abe
# ╠═0d1164df-8236-494b-b8b9-71481c94c0d9
# ╠═942f314e-927b-4371-8c83-83801c860b4d
# ╟─6f75889b-1c7f-4261-bf27-7c991ee9e414
# ╠═6016a736-11da-4451-aa82-cc3045e782db
# ╠═74290eff-781b-44c9-8a90-96bffbe040df
# ╠═876fa74c-9c30-4f0b-9a5b-82bb6597cd47
# ╠═34bad558-e70f-4d46-a9ab-7acc6c89db7a
# ╟─06e8320c-ddd9-4d13-bca3-10fb5c3fb7ad
# ╟─ca1f0910-d417-41bc-ae2d-eebec7f3e1e9
# ╠═b0cdc9d6-738a-4583-b821-052ada846d39
# ╠═3b7fe147-ea94-422f-a943-6f3bd577edf1
# ╠═fc64996f-54ba-4c7a-8cfb-21133cec1fbe
# ╟─1f1e9c9b-e5fa-41b1-852f-cadad703ee4b
# ╠═e3dbe2f5-7e97-45b7-9b75-1acaf8f1031b
# ╟─39414e5e-1256-4497-9738-e2ecdff62d9d
# ╠═7fa55e99-5c0c-465d-a879-bd844e516131
# ╠═06dee467-1f54-48e1-908d-8e4c9028a748
# ╠═aa7b8b58-f959-47de-84d7-8c9cf3ad96be
# ╟─6fb68c61-1ef0-4efc-bcbc-dd9d219c3ebb
# ╟─a9163072-cad9-4b0b-b154-d315c6b68de4
# ╟─c50f50f4-84a3-4a81-bacc-b8ce99d6b257
# ╠═f32db22e-d111-4bf5-9989-a698b0d22626
# ╠═efb34c1a-5505-49f1-aa7f-24f6fd1fc01d
# ╠═444e4eba-9b5a-4e37-8853-5d24c5c398ca
# ╟─fe2c7c2b-63e3-4cbf-b432-b028ec599292
# ╠═63d99d6f-addf-4efc-a680-d4c4733e3941
# ╠═a2f40cbe-f46f-44c6-a81f-aae082c27c1a
# ╠═e2890775-2e29-4244-adac-c37f8f2a8a8e
# ╠═645525f1-77cb-4b18-81df-3eafc0b4004e
# ╟─73444608-d0de-440d-8be3-5a02dadcadc7
# ╠═7e532caa-b8a5-43ab-af4a-4590ffd1a9dc
# ╠═2e683999-0d94-47a1-ab62-cd10b8c3b300
# ╠═8ea2e159-ef17-4ddd-b5a7-5f6c8d67238a
# ╠═3fc67210-6df1-481b-8aa4-48865f9f46b5
# ╟─3991be1c-01f3-416a-a841-18f025a97e24
# ╟─ba59678b-1606-4132-9f19-0dae1e660195
# ╠═ed7cb5c1-528a-4356-80dd-b337107eaf1f
# ╟─5b237453-472f-414e-95e0-f44e980ea93a
# ╠═ef43aef6-80ec-4976-91e6-84a74d29a83e
# ╠═f75ad936-8c06-4e00-92d7-1f86532c0072
# ╟─c75744a0-3c3f-4042-a796-6cbd9ec11195
# ╠═2cc52188-b262-4f65-b042-ad94d90523d8
# ╠═4b98bd17-de33-4648-b737-6b175905b2c7
# ╠═b176823e-b8b5-413d-87b1-90d7efa0e377
# ╟─d8983a9d-1880-4dc4-9c17-23281767e0c2
# ╟─ddee1cf7-7977-407d-a004-08b52f6ff8c8
# ╠═485c046d-8329-4541-bd9d-eb180c01bde6
# ╠═b9318fcf-117e-438e-8bb4-985a9372e2d8
# ╟─ba0545e7-c6df-42e2-a9cd-4ecd490d13e8
# ╠═449f1aae-57b4-4b24-98d9-32c514e00821
# ╠═510f07d9-62e0-40a7-b974-e2ae9bad7f73
# ╠═905793d5-93c5-4d86-9a88-33d6d806d88a
# ╠═c17da7e0-d701-45e6-9967-b2cb0d2b057e
# ╠═d9a82c82-576e-4ee5-a8be-aea1332b3e74
# ╠═b77903d1-c788-4efd-80c4-313859e856e5
# ╠═01af74cf-172d-4561-a7d9-6131a22b4161
# ╠═5def5552-fc0e-4e44-bed2-49edd810c75a
# ╠═a2e8fdca-7f23-4f94-8b94-502ff29500bc
# ╠═072bf511-749d-421b-8e56-d706da88f031
# ╠═f6f3c9b7-dd9d-4f7b-9626-93534c15f199
# ╟─318a6643-5377-4152-8468-51dae1b78144
# ╠═bddafce9-30e4-4708-96ae-938bff9edfe7
# ╠═ebb0c754-12f1-4f80-a5f6-98a61b915fa6
# ╠═a5fe54fb-4f92-4a35-8038-8d36a4aa065c
# ╠═35e70aa4-7b87-4730-8e05-f39d29c20a9e
# ╠═4748a526-8d2e-43a6-8f30-82abf238d624
