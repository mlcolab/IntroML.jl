### A Pluto.jl notebook ###
# v0.19.26

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
    using DataFrames,
        Flux,
        LinearAlgebra,
        Latexify,
        LaTeXStrings,
        LogExpFunctions,
        Optim,
        Plots,
        PlutoUI,
        Random,
        Symbolics,
        Turing
    using Turing: Flat
    Plots.theme(:default; msw=0, ms=3, lw=2, framestyle=:grid, guidefontsize=14)
end

# ╔═╡ 2c05ac8e-7e0b-4d28-ad45-24e9c21aa882
md"""
# Supervised learning: One step at a time

This notebook was created for the ML ⇌ Science  Colaboratory's workshop [Introduction to Machine Learning](https://mlcolab.org/resources/introml-july-2022-workshop-materials).

We here focus on [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning).
As we progress from a simple linear regression to a first example of a neural network, we introduce the basic concepts needed to understand machine learning.

!!! note
	This notebook is written using [Pluto](https://github.com/fonsp/Pluto.jl), a reactive notebook environment.
    If you're viewing this on the web and the interactive elements don't work for you, click "Edit or run this notebook" on the top right to find instructions for making it fully interactive.
    Note that because we load several heavy dependencies, the Binder option will likely not work.
"""

# ╔═╡ 31b1a0e4-216f-4c90-b16e-f542000c8aee
md"""
It's useful to explore our data by plotting.

!!! question
	What would you guess ``f`` looks like?
    How would your guess change if data points appeared in one of the white spaces?
"""

# ╔═╡ 21a16ac9-f83e-46e8-81a2-09f79aa17eae
md"""
!!! note
	For this model, this particular error function is proportional to the sample variance of ``y``, so the value of ``w_0`` that minimizes the error (i.e. variance) is the sample mean of ``y``, that is, ``w_0 = \overline{y}``!

    In other words, the mean observation is the most useful single-parameter model of a dataset (for this error measure).
    When an article reports the sample mean of some observations, one way to interpret that is as a single-parameter model that models any and all observations as independent of any known or unknown features.
    This is a learned model, but not one a very exciting one.
"""

# ╔═╡ 67486053-5971-406b-8680-0d80803d797a
line_trace_manual = [
    [0.3, -3],
    [2.704, -3],
    [2.704, -1.94],
    [2.243, -1.94],
    [2.243, -0.93],
    [1.577, -0.93],
    [1.577, -0.16],
    [1.266, -0.16],
    [1.266, 0.27],
    [1.101, 0.27],
    [1.101, 0.5],
    [0.96, 0.5],
]

# ╔═╡ d1cd6a46-dc5b-4188-a891-703b50bce186
let
    p = plot(; size=(550, 500))
    plot!(p, first.(line_trace_manual), last.(line_trace_manual); color=:blue)
    scatter!(
        p, Base.vect.(line_trace_manual[begin])...; color=:black, marker=:rtriangle, ms=6
    )
    plot!(
        p;
        xlims=(-2.1, 4.1),
        ylims=(-3.1, 2.1),
        aspect_ratio=1,
        xlabel=L"w_0",
        ylabel=L"w_1",
        legend=false,
    )
    scatter!(
        p,
        Base.vect.(line_trace_manual[end])...;
        color=:black,
        marker=:square,
        ms=4,
        label="",
    )
end

# ╔═╡ 7f9e91b8-ee23-4d73-bfe5-c58a29b77abe
md"""
### Making the computer learn step-by-step

There are two problems with how we have been fitting parameters so far.
First, it's manual.
This won't scale to more than a few parameters.
Second, coordinate descent fits one parameter at a time, when ideally we would fit all at the same time (this is like taking diagonal shortcuts in the plot above).

The good news is that we can automate this process!
"""

# ╔═╡ 9049eeca-0db8-41d2-93ee-e0b4e445c9fd
md"""
We can think of ``g`` as defining a "vocabulary" of simple functions that can be used as components to construct our model.
A large vocabulary of functions enables the model to be more expressive.
"""

# ╔═╡ 06e8320c-ddd9-4d13-bca3-10fb5c3fb7ad
md"""
This approach of manually selecting features works well when:
1. we already can guess quite a lot about the function from domain expertise and intuition without looking at the data
2. the function is simple enough that a very small number of features can be linearly combined to approximate it
3. the data is small and simple enough that we can plot it and guess what features might be useful.

Often in machine learning applications, we're not so lucky, so we need generic approaches to construct useful computed features.
"""

# ╔═╡ 1f1e9c9b-e5fa-41b1-852f-cadad703ee4b
md"""
### Choosing the right model

We see that for low ``n``, we don't fit the data very well, but by ``n=3`` we get quite a good fit.
By the time we reach ``n=9``, the model has the same number of free parameters as we have data points, so it's able to perfectly hit every data point and reach an error of 0.
As we increase ``n`` even more, the values of the function between the data points become more extreme.

Here are the trained models for 4 selected values of ``n``.

!!! question
	Which model is best?
"""

# ╔═╡ 267532b6-ba74-4e74-992a-6cabb03949a0
md"""
While at first, the error decreases for both the training and validation data, eventually validation error increases, while training error still decreases.
The validation error is minimized for ``n=3``, so we might call models with ``n<3`` underfitted and models with ``n>3`` as overfitted.

Let's see what happens with 5 times the amount of data.

!!! question
	Do you see any different behavior as ``n`` increases beyond 9?
"""

# ╔═╡ 3996d09f-c115-4097-b994-6f3f573912fc
md"""
The difference in training and validation error is less extreme for higher values of ``n``, and in fact, ``n=5`` is where we now minimize validation error.
In general, as we collect more data, we have more information about the form of ``f``, so we can fit more complex models with less risk of overfitting.

There are more sophisticated approaches we could take, including dividing the data into multiple sets of training and validation sets.
This is called [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), and we won't discuss it further here.
"""

# ╔═╡ 6fb68c61-1ef0-4efc-bcbc-dd9d219c3ebb
md"""
## Regularization

It may seem a little strange that adding more weights can make the model worse, since every simpler model is contained within all the more complex ones (e.g. the model ``w_0 + w_1 x`` can be written as ``w_0 + w_1 x + w_2 x^2`` just by setting ``w_2=0``).
One way to understand this is to look at the fitted values of the weights as ``n`` increases.

!!! question
	What happens to the magnitude of the weights with increasing ``n``?
"""

# ╔═╡ a2466cba-65ea-41a4-b222-c397614453b2
md"""
Think for a moment about the effect of ``\lambda``.
We increase its value because we *a priori* expect the wild oscillations to be unreasonable.
That is, ``\lambda`` is one way to incorporate prior information (i.e. domain expertise or intuition) about smoothness into our model.

Wouldn't it be nice if there was a principled way to do that?
"""

# ╔═╡ a1671960-9b0b-47f2-8d3a-74d67a122ce0
md"""
In regression models, the weights are interpretable, but suppose we are not primarily interested in ``w`` but rather the function ``\hat{f}`` they parametrize.
Since each sampled ``w`` defines a ``\hat{f}``, we could visualize the same distribution by plotting an ensemble of curves.

We can think of each curve as a hypothesis, and the variability in the ensemble reflects the uncertainty we have about which hypothesis is most consistent with all of the information we have used.
Again, we plot the single best fitting curve with a white border.
"""

# ╔═╡ df8f0019-84b9-4309-ab16-4a909fa94e88
md"""
## Families of models

We saw earlier the general strategy of adding computed features ``g`` to increase model complexity.
We can also use this approach to encode model constraints.
For example, if we know the function ``f`` is periodic with period ``P`` with respect to ``x``, then we could encode this constraint in the model by choosing ``g`` to be sine and cosine functions of increasing frequency:

!!! model "Model: Fourier series"
	```math
	\hat{f}(x_i; w) = w_0 + \sum_{j=1}^n w_j\cos\left(\frac{2\pi j}{P} x_i\right) + w_{j+1} \sin\left(\frac{2\pi j}{P} x_i\right)
	```

This is called a [Fourier series](https://en.wikipedia.org/wiki/Fourier_series) and is essentially a polynomial for periodic functions.
We could even add aperiodic terms to the sum to explicitly model periodic and aperiodic effects.

Let's look at this vocabulary of functions:
"""

# ╔═╡ d4c08329-baa4-438a-9555-ee0ea5443e44
let
    funs = vcat(one, reduce(vcat, [[x -> cospi(2j * x), x -> sinpi(2j * x)] for j in 1:4]))
    plots = map(funs) do gj
        lex = latexify(gj(Symbolics.Sym{Real}(:x)); env=:raw)
        plot(gj, 0, 1; color=:orange, xlabel=L"x", ylabel=L"%$lex", label="")
    end
    plot(plots...; link=:both)
end

# ╔═╡ 2909ff18-e98b-4149-843f-1c4709cfbb37
let
    funs = [exp, x -> -exp(x), tanh]
    plots = map(funs) do f
        lex = latexify(f(Symbolics.Sym{Real}(:z)); env=:raw)
        plot(f; xlabel=L"z", ylabel=L"%$lex", label="")
    end
    plot(plots...; link=:both)
end

# ╔═╡ dfebe8a3-fbe5-4381-bce5-1cd403a7b365
plot(
    [atan tanh logistic x -> clamp(x, -1, 1)];
    layout=(2, 2),
    legend=false,
    title=[L"\arctan(z)" L"\tanh(z)" L"\mathrm{logistic}(z)" L"\mathrm{piecewise\ linear}"],
    xlabel=L"z",
    ylabel=L"p(c=1)",
)

# ╔═╡ 24b1008e-f038-4d3d-a7f0-43d4488387f4
md"""
See how where there is a clear separation of points from the two classes, the predicted probability sharply transitions from 0 to 1, but where there's more overlap, it more gradually transitions, so that there's a wider region where the probabilities are not close to 0 or 1.
"""

# ╔═╡ 3316bd55-0f83-48d1-8512-f9192953d716
md"""
## A quick $(html"<s>costume</s>") notation change

At some point writing out sums is tedious, and it's more convenient to work with matrix/vector notation.
This new notation also makes evident why ML is so dependent on [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra).
Let's slightly abuse notation by defining ``\hat{f}(x)`` as ``\hat{f}`` applied elementwise to each data point (row) in ``x``:

```math
\hat{f}(x) = \begin{pmatrix}\hat{f}(x_1) \\ \hat{f}(x_2) \\ \vdots \\ \hat{f}(x_m) \end{pmatrix}
```

Similarly, let's define ``g(x)`` as a function that for each data point (row) of ``x`` computes a row vector with ``n`` entries, each corresponding to a computed feature:

```math
g(x) = \begin{pmatrix}
	g_1(x_1) & g_2(x_1) & \dots & g_n(x_1)\\
	g_1(x_2) & g_2(x_2) & \dots & g_n(x_2)\\
	\vdots & \vdots & \ddots & \vdots \\
	g_1(x_m) & g_2(x_m) & \dots & g_n(x_m)
\end{pmatrix}
```

Then we can write our model as a matrix multiplication:
```math
\hat{f}(x) = g(x) w.
```

This particular notation is convenient as we progress to neural networks.
"""

# ╔═╡ 6f2399db-24e7-4586-a93e-bdb38279470f
md"""
With about 3 or 4 hidden units, which corresponds to 10 or 15 parameters, we're able to model what we took 4 parameters to model with logistic regression.
We potentially could reduce the number of parameters by spreading these units across more hidden layers, since nesting functions makes the model more expressive.
"""

# ╔═╡ ea518070-cc7d-4a33-b9fd-082f7f1aeca1
md"""
### Fitting 2D data with neural networks

Now that we have a generic framework for constructing neural networks, we can do the same thing for arbitrary numbers of features.
When we have two-dimensional data, such as points on a plane, we have two features.
For example, we may use water amount ``x[1]`` and total sunlight hours ``x[2]`` to predict whether a plant will be dead (``c=0``) or alive (``c=1``) after three months.

Instead of switching our main interest to botany, we generate a new synthetic dataset.
"""

# ╔═╡ cc67a1d7-f195-4deb-9b39-22e06a75283d
md"""
!!! question
	Could a logistic regression model fit this data?
"""

# ╔═╡ a96569a4-4ab2-4681-ab4f-fae744a0a671
md"""
## Wrapping it up

To recap, we explored approaches for adding and evaluating model complexity, which led us first to linear regression, then polynomial regression, then logistic regression, and finally to simple neural networks.

Now that you've built some intuition and familiarity with the concepts, we recommend trying out [TensorFlow Playground](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0.001&noise=0&networkShape=5&seed=0.54504&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)!
"""

# ╔═╡ 5b237453-472f-414e-95e0-f44e980ea93a
md"""
# Utilities

This section contains data generation, utility functions and UI elements used in the above notebook.
"""

# ╔═╡ f75ad936-8c06-4e00-92d7-1f86532c0072
TableOfContents()

# ╔═╡ 2cf7fd33-2fda-4311-b699-c8441181b292
@register_symbolic Flux.σ(x)

# ╔═╡ 54ef467e-8cc2-436e-bddc-37c57b6e6980
html"""
<style type="text/css">
@media (prefers-color-scheme: light) {
	pluto-output div.admonition.model {
		border-color: #dba058;
	    background: #f7d698;
	}

	pluto-output div.admonition.model .admonition-title {
		background: #dba058;
	}
}

@media (prefers-color-scheme: dark) {
	pluto-output div.admonition.model {
		border-color: #d7ae61;
	    background: #7d541f;
	}

	pluto-output div.admonition.model .admonition-title {
		background: #d7ae61;
	}
}

</style>
"""

# ╔═╡ b176823e-b8b5-413d-87b1-90d7efa0e377
important(text) = HTML("""<span style="color:#d28800"><strong>$text</strong></span>""")

# ╔═╡ 18198312-54c3-4b4f-b865-3a6a775ce483
md"""
## The data

We are going to work with a collection of ``m=10`` observations or ``y``-values, for example, _the height of a plant_ in meters. For each of them we get as potentially useful information an ``x``-value, for example _the amount of water the plant received over a certain period_, in liters.

In this setting, each individual pair ``(x_i, y_i)`` is called an $(important("instance")) or $(important("example")).
All ``10`` datapoints form our $(important("dataset")) ``\mathcal{D}:=\{(x_i,y_i),\ \ i=1\ldots 10\}``.
Each ``x_i`` is a $(important("feature")) or $(important("attribute")) value and considered an input, while each ``y_i`` is a $(important("label")) or $(important("response")) and considered an output.
"""

# ╔═╡ 435c9f3b-640a-4f54-a836-05fde7ef51a2
md"""
Our goal will be to use this data to learn a relationship between ``x``-values (water) and ``y``-values (height) that $(important("generalizes")), i.e. such that if we are given some new ``x``-value, we can do a decent job at predicting the corresponding ``y``-value.
In mathematical terminology, we seek to find a function ``y=\hat{f}(x)``.
In ML we call this function a $(important("model")), and our goal in ML is to devise ways to learn models automatically from the data -- or $(important("learning algorithms")).
Let's explore how to build ML algorithms together!

!!! note
    No plants were harmed to generate dataset ``\mathcal{D}``.
    Instead of measuring water amounts and plant heights, we generated ``10`` ``x_i`` values at regular intervals and used a function ``f`` (that we keep secret from you) to generate each ``y_i``:
    ```math
    y_i = f(x_i) + \mathrm{noise}.
    ```
    These "invented" outputs ``f(x_i)`` also received some noise to reflect measurement errors.
"""

# ╔═╡ 48f03e74-1e25-41b8-a21c-fd810fadc2cf
md"""
## Building a model

In supervised learning, our goal is to build a function that approximates the unknown ``f``.
We'll call this function ``\hat{f}`` (read "``f``-hat").
``\hat{f}`` is a $(important("model")).
A model is just a function!

In machine learning, when we evaluate ``\hat{f}`` on the raw features ``x``, the result is called $(important("inference")) or $(important("prediction")).
Note that this is different from how the term "inference" is used in statistics!

The first step to building any model is choosing its structure, i.e. choosing the equation of the function.
"""

# ╔═╡ def60ead-a40b-4376-82cd-a77455f6b942
md"""
This process of fitting one coordinate at a time to minimize the loss is called $(important("coordinate descent")).
Here I've saved a trajectory I took manually performing coordinate descent.
I stopped early because I was tired of manually tweaking.
"""

# ╔═╡ 6f75889b-1c7f-4261-bf27-7c991ee9e414
md"""
We can think of the plot above as a topographical map of some landscape, where we want to find the lowest point.
Each step the computer takes is in a direction perpendicular to the level curve of the error function at that point.
This is the direction of steepest descent, which is the negative of the derivative of the error function with respect to the weights (i.e. the gradient) at that point.

For any set of weights (and almost any model), a modern machine learning package can automatically compute this direction using a method called [$(important("backpropagation"))](https://en.wikipedia.org/wiki/Backpropagation) (AKA reverse-mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)), where it approximates how much each parameter is responsible for the error and should therefore change to minimize the error.

This isn't enough though to train the model.
We also need to know how much to change each weight (i.e., how far to step).
It seems intuitive that early on we might need larger steps to cover more terrain faster but in the end want to take smaller steps to avoid stepping too far.

This is a really simple landscape with a single large valley, but often in ML, the landscape can be very bumpy, with lots of tiny hills and valleys.
Finding the global minimum in such cases is often not possible, and finding a low point at all requires a sophisticated algorithm for searching the parameters, using as one component the gradient.
Modern ML frameworks contain many such training algorithms.

From now on, we will ignore backpropagation and algorithm used to fit the parameters and treat them together as a black box.
"""

# ╔═╡ 39414e5e-1256-4497-9738-e2ecdff62d9d
md"""
Again, to answer the question, we need to choose a notion of "bestness."

The goal is to have a model that not only fits the data well but has also learned something about ``f``, which we can only access through observations or prior information.
That is, we want a model that also generalizes well to data that we didn't use to fit it.

We say that a model $(important("underfits")) if it does a poor job predicting the training data.
Conversely, we say that a model $(important("overfits")) the training data if it fits the training data well but held-out data poorly.

To check this for our models, we hold out a $(important("validation set")) ``(x_\mathrm{val}, y_\mathrm{val})``, data similar to ``(x, y)`` that we only use to validate our model.

Often for validation, we use a different error function than for training.
Here, we use the $(important("root-mean-squared error")):
```math
E_\mathrm{RMS}(w) = \sqrt{\frac{2E(w)}{n}}
```

!!! question
	What do you observe as we increase the maximum order ``n``?
"""

# ╔═╡ fe2c7c2b-63e3-4cbf-b432-b028ec599292
md"""
See how even lower order weights become larger in magnitude as the maximum order increases.
If we were to look at weights for ``n > 9``, they continue to explode in magnitude.

Ideally, we would have an approach that avoids overfitting regardless of the amount of data being used.
We can in fact do this by $(important("regularizing")) the weights.
In effect, we add a term to our loss function that penalizes large weight values.
Our regularized error function is

```math
\tilde{E}(w) = E(w) + \frac{\lambda}{2}\sum_{j=1}^n w_j^2.
```

The strength of the penalization is controlled by the magnitude of ``\lambda``.
When ``\lambda=0``, we apply no regularization, and the weights are fit only by the data.
When ``\lambda`` is very large, the regularization term dominates, and it takes very strong signal in the data to move the weights away from 0.

!!! question
	How does the fit model look different as you increase ``\lambda``? How does the error as a function of ``n`` change?
"""

# ╔═╡ 73444608-d0de-440d-8be3-5a02dadcadc7
md"""
## Reflecting uncertainty in the learned model

All along we've been showing the single best approximation ``\hat{f}``, but data is noisy and incomplete, so this is somewhat overconfident an estimate of ``f``.
How overconfident?
We don't know, as our model contains no measure of uncertainty of fit.

Regression models are statistical models, and specifically, we can write them as Bayesian models.
In a $(important("Bayesian")) model (AKA a "probabilistic model"), uncertainties are quantified with probabilities.

Bayesian models combine a likelihood function (i.e. our error function) with a prior on the model parameters (i.e. our regularization term), which define together a joint distribution of parameters (i.e. our weights) given the data, called a $(important("posterior distribution")).

For our linear regression model, with 2 degrees of freedom, we can visualize draws from this distribution (points) versus the single best fit (point with white border).
"""

# ╔═╡ b0773555-44ac-4b06-a410-d25ee1f42399
md"""
Interestingly, our overparametrized model has quite a wide ensemble, since many possible curves are able to almost pass through our training data.
Similarly, our underparametrized models likewise have a wider ensemble, as many linear fits have similar errors, while for ``n=3``, the ensemble is quite tight.

Bayesian models are very useful for certain problems and have distinct advantages and disadvantages.

Advantages:
- We can incorporate prior information in a principled way (i.e. we have rules for setting the amount of regularization)
- Bayesian models are typically more robust against overfitting.
- We can represent uncertainty on hypotheses via a joint distribution.
- We can draw samples for visualizing uncertainty, checking assumptions, and propagating uncertainty to downstream analyses and decisions (generative modeling).
- There are many methods available for model checking.
- Often (but not always), Bayesian models are interpretable.

Disadvantages:
- Difficult to scale to large data or many parameters, since drawing samples is expensive.

The field of machine learning devoted to the construction of and computation with Bayesian models is called $(important("probabilistic programming")).
The key tools here are probabilistic programming languages (PPLs).
PPLs generally provide a convenient syntax for writing models, a library of robust samplers for drawing samples given the model, and usually plots and diagnostics for interpreting and improving on the models.

If data is scarce, uncertainty quantification is important, and/or you want interpretable models, you may want to try out probabilistic programming.
"""

# ╔═╡ 954f9925-3346-49b6-8446-264e617efabc
md"""
Encoding constraints in the model is called $(important("inductive bias")).
Without inductive bias, it might take enormous amounts of data and a very complex model to learn the constraints that we already know; by encoding the constraint, we are able to learn more with the data we already have.

It's also useful to be able to encode constraints on the labels.
The approach we have taken so far works when the labels can occur anywhere on the real line, but even 1-dimensional labels can have many constraints/properties, including:
- positive
- negative
- bounded

We can encode these constraints by applying a function ``\phi`` to the output of the model

!!! model "Model: linear combination of functions with link function"
	```math
	\hat{f}(x_i; w) = \phi\left(\sum_{j=1}^n w_j g_j(x_i)\right),
	```

where ``\phi`` smoothly maps from the entire real line to some subset of the real numbers.

Here are 3 useful functions that transform the real line to the positive numbers, negative numbers, or an interval:
"""

# ╔═╡ ed7cb5c1-528a-4356-80dd-b337107eaf1f
md"""
## Hijacking regression for classification

Suppose some expert looked at our ``y`` values and assigned a label to each depending on whether they were above or below some threshold ``t``.
Since ``y`` is noisy, the expert may be less certain how to assign values close to ``t``.
So they could encode the label as a probability ``p(y > t)``.
Such a label would be continuous and bounded between 0 and 1.
Then they may assign a less certain probability like ``p(y > t) = \frac{1}{2}`` for points near the threshold and either much higher or lower probabilities for values far from the threshold.
Modeling these probabilities would be a different regression problem.

Alternatively, for convenience they might ignore the uncertainty and assign discrete, binary labels ``c=1`` for ``y > t`` and ``c=0`` for ``y \le t``.
Since ``c`` is discrete, building a model to predict ``c`` is a $(important("classification problem")).

Drag the slider below to change ``t`` to get an intuition for how the continuous and discretized labels relate.
"""

# ╔═╡ ba59678b-1606-4132-9f19-0dae1e660195
md"""
We could use any of our previous approaches to model probabilities.
However, we know that probabilities must be between 0 and 1, so it's better to encode that information into the model.
We do this by applying a $(important("sigmoid"))al (S-shaped) function to the output.

Below are various common sigmoids we can use.
Besides their output domain, which can be harmonized through scaling and shifting, they have subtly different properties that we won't cover.
"""

# ╔═╡ b53798f9-24c2-4def-ab6f-447a5d809865
md"""
When predicting probabilities, it's common to choose the logistic function, which we will refer to only as ``\sigma``.

Our model is then
!!! model "Model: logistic regression"
	```math
	\hat{f}(x_i; w) = \sigma\left( \sum_{j=1}^n w_j g_j(x_i)\right).
	```

Suppose we interpret the discrete class labels as absolutely confident probabilities, i.e. ``c=1`` could be interpreted as ``p(y > t) = 1``, while ``c=0`` could be interpreted as ``p(y > t) = 0``.
In this case, we could use the exact same model to perform classification, but instead of predicting binary classes it predicts probabilities of assignment to each class.
These probabilities are more useful than the class labels themselves, because we can more easily identify when our model has a hard time making a prediction, and we can always use thresholding to turn our probabilities into class predictions.

While we could use the same sum-of-squares loss on this function, it turns out that a more appropriate loss is a $(important("cross-entropy loss")):

````math
E(w) = -\sum_{i=1}^k c_i \log \hat{f}(x_i; w) + (1 - c_i) \log (1 - \hat{f}(x_i; w))
````

This model with this choice of loss is called [$(important("logistic regression"))](https://en.wikipedia.org/wiki/Logistic_regression).

Below we show the results of fitting an order ``n=33`` polynomial with logistic function output to the binary labels.
"""

# ╔═╡ 5505fc32-1e46-4256-831c-d1b94d1e946c
md"""
## Nesting functions for model expansion

To recap, we've seen 3 strategies for model expansion:
1. Applying a function to the inputs to compute features
2. Applying a function to the outputs to change the output domain
3. Doing both at the same time

We've so far used all of the weights in the middle, but why not apply some weights, then a function, then apply more weights?
That is, why not have two matrices of weights ``W_1`` and ``W_2``:

!!! model "Model: feed-forward neural network with 1 hidden layer"
	```math
	\hat{f}(x; W_1, W_2) = \sigma(\sigma(g(x) W_1) W_2).
	```

There's no good reason for us not to try this.
In fact, it turns out that alternating application of linear operators (i.e. weighting and adding) and $(important("activation")) functions like our sigmoid is incredibly expressive.
These are called $(important("neural networks")).
"""

# ╔═╡ 94a9846b-ff01-487d-aeac-ddd4ab81610c
md"""
## Neural Networks

One example of a neural network for classification is

!!! model "Model: feed-forward neural network with 1 hidden layer"
	```math
	\hat{f}(x; W_1, W_2) = \sigma(\sigma(x W_1) W_2).
	```

There's a really intuitive way to interpret this.
Let ``g(x; W_1) = \sigma(x W_1)``.
Then ``\hat{f}(x; W_1, W_2) = \sigma(g(x; W_1) W_2)``.
So this neural network is effectively a logistic regression with weight-dependent computed features.
Training the network then also involves learning computed features that are useful for minimizing the error.
This is a very useful way to understand neural networks!

It is more common to represent neural networks graphically:
$(PlutoUI.Resource("https://svgshare.com/i/fmy.svg", :width=>450))

Each node in the network is called a $(important("unit")) or $(important("neuron")) and corresponds to a single input feature, computed feature, or output label.
Each edge is a single weight in a weight matrix.

The data progresses through different stages, called $(important("layers")).
The first layer is where features are input to the model.
These can be raw features; the subsequent layers refine suitable features for the output layer at the end to match the training data.
Layers between input and output are termed $(important("hidden layers"))
"""

# ╔═╡ c75744a0-3c3f-4042-a796-6cbd9ec11195
md"""
## UI elements

This section contains UI elements and variables they are bound to.
"""

# ╔═╡ 879e4355-d4f0-492c-9ac2-7359c794a9fd
w0_input = @bind w0 Scrubbable(-3:0.1:3; default=0.3, format="0.1f");

# ╔═╡ cd49e0a5-4120-481a-965e-72e7bdaf867c
md"""
## The proverbial straight line

### The horizontal line

The simplest model we could choose for ``f`` is a horizontal line:

!!! model "Model: horizontal line"
	````math
	\hat{f}(x_i; w) = w_0
	````

This model has one free parameter ``w_0``.
Written this way, this is called an $(important("untrained model")).
Once we fix the value of ``w_0``, it is called a $(important("trained model")).

Training is just the process of setting ``w_0`` to achieve some goal.

Drag the below number left or right to adjust ``w_0``.

``w_0=`` $w0_input

!!! question
	Which fit is "best"?
"""

# ╔═╡ 4c7e53c5-1271-4acf-95cc-5345564d1b15
md"""
To say which fit is best, we need a measure of goodness of fit, or equivalently, of badness of fit.
That is, we need a notion of error; the best fit has the smallest error.
This is called an $(important("error function")) or $(important("loss function")).
Here, we choose a "sum-of-squares" error function:
````math
E(w) = \frac{1}{2}\sum_{i=1}^m (\hat{f}(x_i; w) - y_i)^2
````

``E(w) = 0`` when ``\hat{f}`` is perfectly able to predict each ``y_i`` from its feature ``x_i``.

Now drag the value of ``w_0`` until you've minimized the error shown at the top of the below plot.

``w_0 = `` $w0_input
"""

# ╔═╡ b8ff692c-2ccb-455c-87c2-590177def1b4
w0_input2 = @bind w0_2 Scrubbable(-2:0.1:2; default=0.3, format="0.1f");

# ╔═╡ fa6a31d1-f1b2-489b-9fef-4f92b5542c80
w1_input = @bind w1 Scrubbable(-1:0.1:3; default=-1, format="0.1f");

# ╔═╡ ae5d8669-f4c4-4b55-9af9-8488e43bcb6c
md"""
### A line with slope

For a better fit, let's give the line not only an intercept ``w_0`` but also also a slope ``w_1``:

!!! model "Model: line"
	````math
	\hat{f}(x_i; w) = w_0 + w_1 x_i
	````

This model now has two degrees of freedom, which we can arrange in a $(important("weight")) vector ``w = \begin{pmatrix}w_0 \\ w_1\end{pmatrix}``.

Weights are one common type of $(important("internal parameter")) of ML models.
In this notebook, all internal parameters can be interpreted as weights.

This model is an example of [$(important("linear regression"))](https://en.wikipedia.org/wiki/Linear_regression).

``w_0`` is often called an $(important("intercept")) or $(important("bias")) term; it shifts the output of the function vertically.

Drag the below values to minimize the loss.
How low can you get the error?

``w^\top = (`` $w0_input2 ``, `` $w1_input ``)``
"""

# ╔═╡ b657b5fe-af35-46e9-93c7-f897e7b22ddc
md"""
Because we now have two weights, our parameter space is 2-dimensional.
So we can equivalently plot the combination of weights on a 2D grid with a readout of the error and just move the values around until the error is minimized:

``w^\top = (`` $w0_input2 ``, `` $w1_input ``)``
"""

# ╔═╡ 0103f249-52e2-4403-b62e-b3ad21536d37
g_options = [
    one => "1",
    identity => "x",
    (x -> x^2) => "x²",
    sin => "sin(x)",
    cos => "cos(x)",
    tan => "tan(x)",
    exp => "exp(x)",
];

# ╔═╡ bf50534d-1c9a-439a-9922-262f64b83c1d
let
    plots = map(g_options) do (gj, _)
        lex = latexify(gj(Symbolics.Sym{Real}(:x)); env=:raw)
        plot(gj, 0, 1; color=:orange, xlabel=L"x", ylabel=L"%$lex", label="")
    end
    plot(plots...; link=:both)
end

# ╔═╡ e06d0925-585f-42e9-9dea-49044625d3d8
g_input = @bind g MultiSelect(g_options, size=5, default=Function[one]);

# ╔═╡ 74290eff-781b-44c9-8a90-96bffbe040df
md"""
## Model comparison

_This section is inspired by Section 1.1 of [Pattern recognition and machine learning](https://link.springer.com/book/9780387310732) by Christopher Bishop._

### Scaling the model up by trial-and-error

Training the model gave us the best fitting line, but we think we can do better.
To do this, we need to allow ``\hat{f}`` to take more complex shapes.

One way we can do this is by defining ``\hat{f}`` as the weighted sum of simpler functions ``g_j``:

!!! model "Model: linear combination of functions"
	```math
	\hat{f}(x; w) = w_1 g_1(x) + w_2 g_2(x) + \ldots + w_n g_n(x) = \sum_{j=1}^n w_j g_j(x).
	```

So far, we have been working with ``x`` only, and this is what we may call a $(important("raw feature")), because we did not process it in any way -- it comes straight from the data.
However,  ``g_j(x)`` are calculated from ``x``, and so we now have ``n`` $(important("computed features")) from our initial raw feature.
Note that if ``g_j`` is a nonlinear function of ``x``, then ``\hat{f}`` is a nonlinear function of ``x`` but still a linear function of the weight vector ``w``.

!!! note
	From here on out, we use the slightly abused notation that a scalar function ``g_j`` applied to an array ``x`` is applied separately to each element ``x_i``.

Let's try selecting some useful scalar functions to add to ``g`` below.

Hint: ``\begin{bmatrix}1 \\ x\end{bmatrix}`` is equivalent to fitting the line with slope.

``g(x) = `` $g_input
"""

# ╔═╡ aa8c6266-0338-44a9-b14b-088f98b04db3
max_order_input = @bind max_order Scrubbable(0:20; default=0);

# ╔═╡ ca1f0910-d417-41bc-ae2d-eebec7f3e1e9
md"""
### Systematically scaling the model by adding terms

One way we can systematically scale the model complexity is by choosing as many computed features as we want from some indexed family of functions ``g_j``.

For example, we can let ``g_j(x_i) = x_i^j``, so that

!!! model "Model: polynomial"
	```math
	\hat{f}(x_i; w) = w_0 + w_1 x_i + w_2 x_i^2 + \ldots + w_n x_i^n = \sum_{j=0}^n w_j x_i^j
	```

Functions of the form of ``\hat{f}`` are called [polynomials](https://en.wikipedia.org/wiki/Polynomial), and many functions (specifically, [analytic](https://en.wikipedia.org/wiki/Analytic_function) functions) can be exactly computed with infinite terms (i.e. ``n \to \infty``) or approximated with finite terms (by picking some manageable ``n``).
Fitting these polynomial models is called [$(important("polynomial regression"))](https://en.wikipedia.org/wiki/Polynomial_regression).


Our line with two degrees of freedom is the special case ``n=1``, though note again that here ``\hat{f}`` is still linear with respect to the weights ``w`` (i.e. this is still _linear_ regression)

!!! question
	What happens as we increase the maximum order ``n``?

``n = `` $max_order_input
"""

# ╔═╡ 06dee467-1f54-48e1-908d-8e4c9028a748
md"``n = `` $max_order_input"

# ╔═╡ efb34c1a-5505-49f1-aa7f-24f6fd1fc01d
md"``n = `` $max_order_input"

# ╔═╡ 9b96ba97-1fe6-4d08-aa69-07a04219686b
max_order_reg_input = @bind max_order_reg Scrubbable(0:20; default=0);

# ╔═╡ c6218e7c-1999-4e90-8868-a81717837fb2
show_contour_input = @bind show_contour CheckBox(; default=false);

# ╔═╡ 2eb005a3-f5b2-4216-b56f-e25157b8c33c
md"""
Let's overlay the computer-generated trajectory on our manual one.

!!! question
	Can you tell what strategy the computer is using to minimize the error?
	What about if you show the contours of the error function?

Show contours of error function: $show_contour_input
"""

# ╔═╡ 186680f8-9b47-4a93-9fa0-81c0c2ea894b
λ_input = @bind λ Scrubbable(exp10.([-Inf; -15:1:0]); default=0, format=".1g");

# ╔═╡ e2890775-2e29-4244-adac-c37f8f2a8a8e
md"``n = `` $max_order_reg_input ``\quad`` ``\lambda =`` $λ_input"

# ╔═╡ fb5e7b56-66e5-45ef-9ad7-b50dd997137d
thresh_input = @bind thresh Scrubbable(-3:0.1:3; default=0);

# ╔═╡ 1e12834c-4b29-41db-ab1f-d93db62c8341
md"``t = `` $thresh_input"

# ╔═╡ 94629920-3a9a-4d50-973f-cd18d3ea50f0
nhidden1_input = @bind nhidden1 Scrubbable(0:10; default=0);

# ╔═╡ d2e5bde1-8c65-494f-8944-b16dec6ab193
md"""
### Fitting 1D data with neural networks

Let's take the simple neural architecture with a single hidden layer displayed above and use it to fit the same classification data we fit in the linear regression example.

!!! question
	How many units does it take before the fit is similar to what we saw with logistic regression?

Number of hidden units = $nhidden1_input
"""

# ╔═╡ 38b63169-2712-48a6-b8b5-66429565cc87
nhidden2_input = @bind nhidden2 Scrubbable(0:10; default=0);

# ╔═╡ 29fb4486-5605-438f-9b1a-a24a19b20c5e
md"""
Sure, we could fit this data with logistic regression if we chose the right computed features.
Importantly, it is *not* possible to fit this data if we only linearly combine raw features (i.e. no line can be drawn to separate the two classes).
But we can also fit this data with a neural net, which would compute the features for us, reusing the same architecture we defined above and only adding a second input feature.

$(PlutoUI.Resource("https://svgshare.com/i/fky.svg", :width=>450))

!!! question
	How many hidden units does it take until this fit looks good to you?

Number of hidden units = $nhidden2_input
"""

# ╔═╡ d8983a9d-1880-4dc4-9c17-23281767e0c2
md"## Definitions"

# ╔═╡ 5e7bda42-0266-4498-906d-9aca8b6c4bf3
f(x) = sinpi(2x) + 2.5x;

# ╔═╡ 805d2824-86cc-45bd-88b0-e6e14d9fde48
f2(x, y) = sinpi(2x) * sinpi(2y);

# ╔═╡ 3f4bffbb-51f4-4446-9b9c-cd3f99edfefa
lossl2(yhat, y) = sum(abs2, yhat .- y) / 2;

# ╔═╡ 04c7209e-c4f8-454b-a883-cb2c5fac5203
error(f, x, y) = lossl2(f.(x), y);

# ╔═╡ 7fa55e99-5c0c-465d-a879-bd844e516131
error_rms(f, x, y; method=error) = sqrt(method(f, x, y) * 2//length(x));

# ╔═╡ ddee1cf7-7977-407d-a004-08b52f6ff8c8
md"## Plotting functions"

# ╔═╡ 485c046d-8329-4541-bd9d-eb180c01bde6
function plot_data!(
    p,
    x,
    y;
    xrange=(0, 1),
    f_hat=nothing,
    f_draws=nothing,
    show_residuals=false,
    data_color=:blue,
    equation=nothing,
)
    if show_residuals && f_hat !== nothing
        sticks!(p, x, y; fillrange=f_hat.(x), color=:red, lw=1)
    end
    scatter!(p, x, y; xlabel=L"x", ylabel=L"y", color=data_color)
    if f_draws !== nothing
        for fi in f_draws
            plot!(p, fi, xrange...; color=:orange, lw=1, label="", alpha=0.2)
        end
    end
    if f_hat !== nothing
        # hack to give the fit a white border
        plot!(p, f_hat, xrange...; color=:white, lw=3.5, label="")
        plot!(p, f_hat, xrange...; color=:orange, label="")
    end
    if equation !== nothing
        xmin, xmax = xlims(p)
        ymin, ymax = ylims(p)
        xmin = xmin + (xmax - xmin) * 0.015
        ymax = ymax - (ymax - ymin) * 0.01
        annotate!(p, [(xmin, ymax, (equation, :left, :top))])
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

# ╔═╡ 510f07d9-62e0-40a7-b974-e2ae9bad7f73
function generate_data(f, x, error; rng=Random.GLOBAL_RNG)
    return f.(x) .+ randn.(rng) .* error
end

# ╔═╡ 905793d5-93c5-4d86-9a88-33d6d806d88a
begin
    error_actual = 0.2
    ndata = 10
    data_seed = 42
    data_test_seed = 63
    x = collect(range(0, 1; length=ndata))
    y = generate_data(f, x, error_actual; rng=MersenneTwister(data_seed))
    ytest = generate_data(f, x, error_actual; rng=MersenneTwister(data_test_seed))
    xmore = collect(range(0, 1; length=50))
    ymore = generate_data(f, xmore, error_actual; rng=MersenneTwister(data_seed))
    ymore_test = generate_data(f, xmore, error_actual; rng=MersenneTwister(data_test_seed))
end;

# ╔═╡ a1ca71d8-2b2c-48de-8088-3cc32135fe5a
round.(DataFrame(; x, y); digits=2)

# ╔═╡ 1b4812d4-3879-4a79-a95d-20cad2959f5c
plot_data(x, y)

# ╔═╡ 942f314e-927b-4371-8c83-83801c860b4d
line_trace = let
    step_length = 0.02
    obj(w) = error(x -> w[1] + w[2] * x, x, y)
    winit = [0.3, -3.0]
    optimizer = Optim.GradientDescent(;
        alphaguess=Optim.LineSearches.InitialStatic(; alpha=step_length),
        linesearch=Optim.LineSearches.Static(),
    )
    options = Optim.Options(; store_trace=true, extended_trace=true)
    res = Optim.optimize(obj, winit, optimizer, options)
    wtrace = Optim.x_trace(res)
end

# ╔═╡ 09877b51-c34c-4351-ab8d-0fbee76d5a1b
let
    c = ymore .> thresh
    data_color = Int.(c)
    p = plot_data(xmore, ymore; data_color)
    hline!(p, [thresh]; color=:black)
    p2 = plot_data(xmore, logistic.((ymore .- thresh) .* 10); data_color)
    plot!(p2; ylabel=L"p(y > t)", ylims=(-0.1, 1.1))
    p3 = plot_data(xmore, c; data_color)
    plot!(p3; ylabel=L"c", ylims=(-0.1, 1.1))
    plot(p, p2, p3; link=:x, layout=(3, 1))
end

# ╔═╡ 90a42425-9f1b-464a-9b10-d0a25cc6717c
let
    max_order = 3
    thresh = 1.2
    c = ymore .> thresh
    features = xmore .^ (0:max_order)'
    function obj(w)
        z = features * w
        # fuse loss and logistic for more numerical stability
        return -sum(c .* z .- log1pexp.(z))
    end
    w = zeros(max_order + 1)
    sol = Optim.optimize(obj, zeros(max_order + 1), LBFGS())
    w = Optim.minimizer(sol)
    f_hat(x::Real) = logistic.(dot(x .^ (0:max_order), w))
    p = plot_data(xmore, c; f_hat, data_color=Int.(c))
    plot!(p; ylabel=L"p(c = 1)")
end

# ╔═╡ 655c8b62-e180-41e6-a3b5-7317cdc76f73
let
    Random.seed!(32)
    c = Int.(ymore .> 1.2)
    data = [(xmore', c')]
    if nhidden1 == 0
        model = Dense(1, 1, Flux.σ)
    else
        model = Chain(
            Dense(1, nhidden1, Flux.σ),  # x -> σ(x * W₁ + b₁)
            Dense(nhidden1, 1, Flux.σ),  # z -> σ(z * W₂ + b₂)
        )
    end

    w = Flux.params(model)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    opt = Flux.Optimise.ADAM()
    for _ in 1:50_000
        Flux.train!(loss, w, data, opt)
    end
    f_hat(x) = only(model(fill(x, 1, 1)))
    plot_data(xmore, c; data_color=c .+ 1, f_hat)
end

# ╔═╡ 10b73d22-cf2a-479c-84b6-6a63a694f398
x2, c2 = let
    rng = MersenneTwister(92)
    x = rand(rng, 400, 2)
    z = f2.(x[:, 1], x[:, 2]) .+ randn.(rng) .* 0.1
    c = Int.(z .> 0)
    x, c
end;

# ╔═╡ 82a245fa-6e02-41f4-bba4-769863a896db
round.(DataFrame("x[1]" => x2[:, 1], "x[2]" => x2[:, 2], "c" => c2); digits=2)

# ╔═╡ dda5a074-3f7b-46dd-bc4b-cb05117ec425
let
    scatter(
        x2[:, 1], x2[:, 2]; group=c2, label=[L"c=0" L"c=1"], xlabel=L"x_1", ylabel=L"x_2"
    )
end

# ╔═╡ f52b2954-6edb-48cc-8dc9-94a4ef613012
let
    Random.seed!(28)
    λ = 1e-6
    data = [(x2', c2')]
    if nhidden2 == 0
        model = Dense(2, 1, Flux.σ)
    else
        model = Chain(
            Dense(2, nhidden2, Flux.σ),  # x -> σ(x * W₁ + w₁₀)
            Dense(nhidden2, 1, Flux.σ),  # z -> σ(z * W₂ + w₂₀)
        )
    end
    w = Flux.params(model)
    penalty() = sum(wi -> sum(abs2, wi), w) * λ
    loss(x, y) = Flux.Losses.mse(model(x), y) + penalty()
    opt = Flux.Optimise.ADAM()
    for _ in 1:20_000
        Flux.train!(loss, w, data, opt)
    end
    f_hat(x, y) = only(model([x, y]))
    lex = latexify(
        f_hat(Symbolics.Sym{Float64}(:x_1), Symbolics.Sym{Float64}(:x_2));
        fmt="%.2f",
        env=:raw,
    )
    equation = L"%$lex"
    p = contourf(
        -0.01:0.01:1.01, -0.01:0.01:1.01, f_hat; color=:coolwarm, lw=0, clim=(0, 1)
    )
    scatter!(p, x2[:, 1], x2[:, 2]; color=c2 .+ 1, group=c2, label=[L"c=0" L"c=1"], msw=0.5)
    annotate!(p, [(0, 1.06, (equation, :left, :top, 10))])
    plot!(; xlabel=L"x_1", ylabel=L"x_2")
end

# ╔═╡ 318a6643-5377-4152-8468-51dae1b78144
md"""
## Models and utilities
"""

# ╔═╡ 4748a526-8d2e-43a6-8f30-82abf238d624
begin
    # compute feature matrix
    function compute_features(g, x::AbstractVector)
        z = similar(x, length(x)..., length(g)...)
        for j in eachindex(g)
            z[:, j] .= g[j].(x)
        end
        return z
    end
    # compute feature vector
    compute_features(g, x) = map(gj -> gj(x), g)
end

# ╔═╡ bddafce9-30e4-4708-96ae-938bff9edfe7
solve_regression(x, y) = x \ y

# ╔═╡ ebb0c754-12f1-4f80-a5f6-98a61b915fa6
solve_regression(x, y, λ) = (x'x + λ * I) \ (x'y) # ridge regression

# ╔═╡ 34bad558-e70f-4d46-a9ab-7acc6c89db7a
let
    w = solve_regression(compute_features(g, x), y)
    wround = round.(w; digits=2)
    f_hat(x) = sum(dot(w, compute_features(g, x)))
    f_hat_sym(x) = sum(wround .* compute_features(g, x))
    err = round(error(f_hat, x, y); digits=2)
    p = plot_data(x, y; f_hat, show_residuals=true)
    lex = latexify(f_hat_sym(Symbolics.Sym{Real}(:x)); env=:raw)
    annotate!(p, [(minimum(x), maximum(y), ("\$\\hat{f}(x)=$lex\$", :left, :top))])
    plot!(p; title="\$E(w) = $err \$")
end

# ╔═╡ 2d730c2f-7320-4879-b6a2-bee8c7c9b338
function as_latex(f; varname=:x, fname="\\hat{f}")
    var = Symbolics.Sym{Real}(varname)
    return L"%$fname(x) = %$(latexify(simplify(f(var)); env=:raw))"
end

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
        return sum(x .^ p .* w)
    end

    fit!(m::PolyModel, x, y) = fit!(m, x, y, identity)

    function fit!(m::PolyModel, x, y, link::typeof(identity))
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

    function as_latex(m::PolyModel; digits=2, kwargs...)
        w = round.(m.w; digits)
        mapprox = PolyModel(w, round(m.λ; digits))
        return as_latex(x -> mapprox(x); kwargs...)
    end

    function Base.show(io::IO, mime::MIME"text/latex", m::PolyModel)
        return show(io, mime, as_latex(m))
    end
end;

# ╔═╡ 72198269-d070-493f-92eb-36135692ca8f
let
    f_hat(x) = w0
    plot_data(x, y; f_hat, show_residuals=true, equation=as_latex(f_hat))
end

# ╔═╡ 288aa7d7-8785-4f55-95e6-409e2ceb203a
let
    f_hat(x) = w0
    loss = round(error(f_hat, x, y); digits=2)
    equation = as_latex(PolyModel([w0]); digits=3)
    p = plot_data(x, y; f_hat, show_residuals=true, equation)
    plot!(p; title="\$ E(w_0)= $loss \$")
end

# ╔═╡ 1a906880-75e1-447b-9adf-31ae44f0135f
begin
    f_hat_line(x) = w0_2 + w1 * x
    error_line = error(f_hat_line, x, y)
    equation_line = as_latex(f_hat_line)
end;

# ╔═╡ 345ae96b-92c2-4ac4-bfdf-302113627ffb
let
    p = plot_data(x, y; f_hat=f_hat_line, show_residuals=true, equation=equation_line)
    plot!(p; title=L"E(w)= %$(round(error_line; digits=2))")
end

# ╔═╡ 0d1164df-8236-494b-b8b9-71481c94c0d9
let
    scatter([w0_2], [w1]; xlims=(-4.1, 4.1), ylims=(-3.1, 2.1), color=:orange, label="")
    plot!(; title=L"E(w)= %$(round(error_line; digits=2))", xlabel=L"w_0", ylabel=L"w_1")
end

# ╔═╡ 6016a736-11da-4451-aa82-cc3045e782db
let
    obj(w) = error(x -> w[1] + w[2] * x, x, y)
    p = plot(; size=(550, 500))
    plot!(
        p,
        first.(line_trace),
        last.(line_trace);
        seriestype=:scatterpath,
        ms=2,
        markercolor=:orange,
        color=:red,
        label="computer",
    )
    plot!(
        p, first.(line_trace_manual), last.(line_trace_manual); color=:blue, label="human"
    )
    contour!(
        p,
        -2.1:0.01:4.1,
        -3.1:0.01:2.1,
        (w0, w1) -> obj([w0, w1]);
        levels=100,
        alpha=show_contour * 0.5,
        lw=1,
        color=:greys,
        colorbar=false,
    )
    scatter!(
        p,
        Base.vect.(line_trace_manual[begin])...;
        color=:black,
        marker=:rtriangle,
        ms=6,
        label="",
    )
    w = fit!(PolyModel(1), x, y).w
    scatter!(p, [w[1]], [w[2]]; color=:black, marker=:square, ms=4, label="")
    scatter!(
        p,
        Base.vect.(line_trace_manual[end])...;
        color=:black,
        marker=:square,
        ms=4,
        label="",
    )
    plot!(
        p;
        xlims=(-2.1, 4.1),
        ylims=(-3.1, 2.1),
        aspect_ratio=1,
        xlabel=L"w_0",
        ylabel=L"w_1",
    )
end

# ╔═╡ fc64996f-54ba-4c7a-8cfb-21133cec1fbe
let
    f_hat = fit!(PolyModel(max_order), x, y)
    err = round(error(f_hat, x, y); digits=2)
    equation = as_latex(f_hat)
    p = plot_data(x, y; f_hat=x -> f_hat(x), show_residuals=true, equation)
    plot!(p; title="\$E(w) = $err \$")
end

# ╔═╡ e3dbe2f5-7e97-45b7-9b75-1acaf8f1031b
let
    plots = map([0, 1, 3, 9]) do max_order
        f_hat = fit!(PolyModel(max_order), x, y)
        p = plot_data(x, y; f_hat=x -> f_hat(x), show_residuals=false)
        plot!(p; title="\$n=$max_order\$")
    end
    plot(plots...; layout=(2, 2), link=:all)
end

# ╔═╡ a9163072-cad9-4b0b-b154-d315c6b68de4
let
    max_orders = [0, 1, 3, 6, 9]
    pairs = map(max_orders) do n
        m = fit!(PolyModel(n), x, y)
        # pad with `missing`s
        "$n" => [round.(m.w; digits=2); fill(md"", maximum(max_orders) - n)]
    end
    DataFrame(pairs)
end

# ╔═╡ 3731fa3f-efb3-43d1-900f-8582cd89d526
function plot_poly_compare(x, y, ytest, max_order; λ=0)
    f_hat = fit!(PolyModel(max_order, λ), x, y)
    p = plot_data(x, ytest; f_hat=x -> f_hat(x), data_color=:magenta, show_residuals=true)
    plot_data!(p, x, y)
    max_orders = 0:max(10, max_order)
    rmse_values = map(max_orders) do n
        m = fit!(PolyModel(n, λ), x, y)
        return error_rms(m, x, y), error_rms(m, x, ytest)
    end
    p2 = plot(max_orders, first.(rmse_values); color=:blue, label="training")
    plot!(p2, max_orders, last.(rmse_values); color=:magenta, label="validation")
    scatter!(
        p2,
        [max_order],
        [rmse_values[max_order + 1]...]';
        color=[:blue :magenta],
        ms=6,
        label="",
    )
    plot!(
        p2;
        legend=:topright,
        xlabel=L"n",
        ylabel=L"E_\mathrm{RMS}(w)",
        xlims=(-0.5, max(max_order, 10.5)),
        ylims=(-0.01, NaN),
    )
    return plot(p, p2)
end

# ╔═╡ aa7b8b58-f959-47de-84d7-8c9cf3ad96be
plot_poly_compare(x, y, ytest, max_order)

# ╔═╡ 2183e956-1b47-4e97-b957-c5df0541ff7b
plot_poly_compare(xmore, ymore, ymore_test, max_order)

# ╔═╡ 9ce43ec1-78f5-414f-9796-ab3159be7985
plot_poly_compare(x, y, ytest, max_order_reg; λ)

# ╔═╡ 86e58e39-186e-470f-832a-32cd86717daa
# for computational reasons, we use the QR parameterization of the regression model
# and specify the priors on the transformed coefficients. For λ=0, this has the same
# MAP solution as the original parameterization, but for λ>0, the MAP solution is
# different.
@model function poly_regress_bayes(
    x,
    y,
    max_order;
    σ=0.2,
    λ=0,
    σw=sqrt(inv(λ)) * σ, # convert regularization to standard deviation
    wprior=iszero(λ) ? Flat() : Normal(0, σw),
    p=0:max_order,
    n=length(x),
    features=x .^ p',
    Q=Matrix(qr(features).Q) * sqrt(n - 1),
    Rinv=inv(qr(features).R / sqrt(n - 1)),
)
    w_tilde ~ filldist(wprior, max_order + 1)
    μ = Q * w_tilde
    y .~ Normal.(μ, σ)
    return (; w=Rinv * w_tilde)
end

# ╔═╡ 0a06b151-461a-470b-927b-851c64d826bf
function draw_samples(m::PolyModel, x, y, ndraws; rng=Random.GLOBAL_RNG, sampler=NUTS())
    mod = poly_regress_bayes(x, y, length(m.w) - 1; λ=m.λ)
    chns = sample(mod, sampler, ndraws)
    params = MCMCChains.get_sections(chns, :parameters)
    weights = generated_quantities(mod, params)
    polys = map(vec(weights)) do nt
        return PolyModel(nt.w, m.λ)
    end
    return polys
end

# ╔═╡ 95f3bbad-1309-40c5-9da2-e1255a325d8b
poly_draws = let
    rng = MersenneTwister(20)
    map([0, 1, 3, 9]) do max_order
        max_order => draw_samples(PolyModel(max_order), x, y, 100; rng)
    end
end

# ╔═╡ d462dc39-f90b-4429-b6b5-7ded05fa3432
let
    draws = Dict(poly_draws)[1]
    obj(w) = error(PolyModel(w), x, y)
    w1 = [m.w[1] for m in draws]
    w2 = [m.w[2] for m in draws]
    p = scatter(w1, w2; color=:orange, alpha=0.25)
    contour!(
        p,
        -5:0.01:5,
        -4:0.01:4,
        (w0, w1) -> obj([w0, w1]);
        levels=100,
        color=:grey,
        colorbar=false,
        lw=1,
    )
    f_hat = fit!(PolyModel(1), x, y)
    scatter!(p, [f_hat.w[1]], [f_hat.w[2]]; color=:orange, msc=:white, msw=1.3, ms=3)
    plot!(
        p;
        xlims=(-4.1, 4.1),
        ylims=(-3.1, 2.1),
        aspect_ratio=1,
        xlabel=L"w_0",
        ylabel=L"w_1",
        legend=false,
    )
end

# ╔═╡ 8ea2e159-ef17-4ddd-b5a7-5f6c8d67238a
let
    plots = map(poly_draws) do (max_order, polys)
        mod = poly_regress_bayes(x, y, max_order)
        f_draws = map(m -> (x -> m(x)), rand(polys, 20))
        f_hat = fit!(PolyModel(max_order), x, y)
        p = plot_data(x, ytest; f_draws, f_hat=x -> f_hat(x), data_color=:magenta)
        plot!(p; title="\$n=$max_order\$")
        p
    end
    plot(plots...; layout=(2, 2), link=:both)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
DataFrames = "~1.3.2"
Flux = "~0.12.9"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.13"
LogExpFunctions = "~0.3.10"
Optim = "~1.6.2"
Plots = "~1.27.4"
PlutoUI = "~0.7.37"
Symbolics = "~4.3.1"
Turing = "~0.21.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "addb13b0aea7688f4fe00d84034f44c777be88f4"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Markdown", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "29e65c331f97db9189ef00a4c7aed8127c2fd2d4"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.27.10"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "b4d13c741b2bbe0e1af305a92d2f105a7caa47af"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.2.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Random", "Setfield", "SparseArrays"]
git-tree-sha1 = "33ea6c6837332395dbf3ba336f273c9f7fcf4db9"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.4"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "0091e2e4d0a7125da0e3ad8c7dbff9171a921461"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.6"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "d7a7dabeaef34e5106cdf6c2ac956e9e3f97f666"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.8"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "9ff1247be1e2aa2e740e84e8c18652bd9d55df22"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.3.8"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "67fcc7d46c26250e89fc62798fbe07b5ee264c6f"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.6"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "01a9f8e6cfc2bfdd01d333f70b8014a04893103c"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.4"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "54b00d1b93791f8e19e31584bd30f2cb6004614b"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.38"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijections]]
git-tree-sha1 = "fe4f8c5ee7f76f2198d5c2a06d3961c249cce7bd"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.4"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "1234b03e94938e6f2b14834dfd3ef45698d5e14f"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "6717cb9a3425ebb7b31ca4f832823615d175f64a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.13.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "8bae903893aeeb429cf732cf1888490b93ecf265"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.49.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "f84967c4497e0e1955f9a582c232b02847c5f589"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "9441451ee712d1aec22edad62db1a9af3dc8d852"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "6c0100a8cf4ed66f66e2039af7cde3357814bad2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5eeb2bd01e5065090ad591a205d8cad432ae6cb6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.93"
weakdeps = ["ChainRulesCore", "DensityInterface"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "c0eb6f717434b3f13cd7a24b412938d952849d77"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.45"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "988e2db482abeb69efc76ae8b6eba2e93805ee70"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.15"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "MacroTools", "OrderedCollections", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "7bc3920ba1e577ad3d7ebac75602ab42b557e28e"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.20.2"

[[deps.DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "8b84876e31fa39479050e2d3395c4b3b210db8b0"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterfaceCore", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4cda4527e990c0cc201286e0a0bfbbce00abcfc2"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.0.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7072f1e3e5a8be51d525d64f63d3ec1287ff2790"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.11"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "04ed1f0029b6b3af88343e439b995141cb0d0b8d"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.17.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "SparseArrays", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "511b7c48eebb602a8f63e7d6c63e25633468dc16"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.10"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "9ade6983c3dbbd492cf5729f865fe030d1541463"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.6.6"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "19d693666a304e8c371798f4900f7435558c7cde"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.17.3"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "bb198ff907228523f3dee1070ceee63b9359b6ab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "659140c9375afa2f685e37c1a0b9c9a60ef56b40"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.7"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "Logging", "MultivariatePolynomials", "Primes", "Random"]
git-tree-sha1 = "47f0f03eddecd7ad59c42b1dd46d5f42916aff63"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.2.11"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "f366daebdfb079fd1fe4e3d560f99a0c892e15bc"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "3f91cd3f56ea48d4d2a75c2a65455c5fc74fa347"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "47be64f040a7ece575c2b5f53ca6da7b548d69f4"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f044a2796a9e18e0531b9b3072b0019a61f264bc"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.17.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "070e4b5b65827f82c16ae0916376cb47377aa1b5"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.18+0"

[[deps.LRUCache]]
git-tree-sha1 = "48c10e3cc27e30de82463c27bef0b8bdbd1dc634"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.4.1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "1cccf6d366e51fbaf80303158d49bb2171acfeee"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.9.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "dfa6c5f2d5a8918dd97c7f1a9ea0de68c2365426"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.5"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random", "Requires", "UnPack"]
git-tree-sha1 = "c3e1189191e4528b605070972d7d4e9cd91dd96b"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "1.0.3"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "c659f7508035a7bdd5102aef2de028ab035f289a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.7.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8b862779314f9299cbf7bdbf2413bcbd9c8e77b2"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.2.6"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c8b7e632d6754a5e36c0d94a4b466a5ba3a30128"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.8.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0f39bc7f71abdff12ead4fc4a7d998fb2f3c171f"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.5"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "eaa98afe2033ffc0629f9d0d83961d66a021dfcc"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.7"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "964cb1a7069723727025ae295408747a0b36a854"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "99e6dbb50d8a96702dc60954569e9fe7291cc55d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.20"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "b05a082b08a3af0e5c576883bc6dfb6513e7e478"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.6"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "b84e17976a40cb2bfe3ae7edb3673a8c630d4f95"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.8"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bc0a748740e8bc5eeb9ea6031e6f050de1fc0ba2"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.6.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6f2dd1cf7a4bbf4f305a0d8750e351cb46dfbe80"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "436c0db5ba99a01452f52a0ab0fd8ad6664d5470"
repo-rev = "ms_nonlazy"
repo-url = "https://github.com/sethaxen/PlutoUI.jl"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "311a2aa90a64076ea0fac2ad7492e914e6feeb81"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "6d7bb727e76147ba18eed998700998e17b8e4911"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.4"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "a5ce741acddc02f0d4fc6505463ca89697d7fb23"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "e961f57a7d5b1d46059b62a8fb7223d6497a5bc4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.15"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "d7d9ebe28062161c1e314ed643097b0c6fe657d9"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.7"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "fe89a8113ea445bcff9ee570077830674babb534"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.81.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "5d2c08cef80c7a3a8ba9ca023031a85c263012c5"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "8982b3607a212b070a5e46eea83eb62b4744ae12"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.25"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "027b43d312f6d52187bb16c2d4f0588ddb8c4bb2"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.19.11"

[[deps.Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "Groebner", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "759d6102719068d95acae0b5480c157fa278ca82"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.3.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "34e6bcf36b9ed5d56489600cf9f3c16843fa2aa2"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.11"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0874c1b5de1b5529b776cfeca3ec0acfada97b1b"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.20"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "631fdfe9720d8c2e2a885b1580c8f8c18347b5da"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.13"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ead6292c02aab389cb29fe64cc9375765ab1e219"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "ebac1ae9f048c669317ad48c9bed815790a468d8"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.61"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─2c05ac8e-7e0b-4d28-ad45-24e9c21aa882
# ╟─18198312-54c3-4b4f-b865-3a6a775ce483
# ╠═a1ca71d8-2b2c-48de-8088-3cc32135fe5a
# ╟─435c9f3b-640a-4f54-a836-05fde7ef51a2
# ╟─31b1a0e4-216f-4c90-b16e-f542000c8aee
# ╠═1b4812d4-3879-4a79-a95d-20cad2959f5c
# ╟─48f03e74-1e25-41b8-a21c-fd810fadc2cf
# ╟─cd49e0a5-4120-481a-965e-72e7bdaf867c
# ╠═72198269-d070-493f-92eb-36135692ca8f
# ╟─4c7e53c5-1271-4acf-95cc-5345564d1b15
# ╠═288aa7d7-8785-4f55-95e6-409e2ceb203a
# ╟─21a16ac9-f83e-46e8-81a2-09f79aa17eae
# ╟─ae5d8669-f4c4-4b55-9af9-8488e43bcb6c
# ╠═345ae96b-92c2-4ac4-bfdf-302113627ffb
# ╠═1a906880-75e1-447b-9adf-31ae44f0135f
# ╟─b657b5fe-af35-46e9-93c7-f897e7b22ddc
# ╠═0d1164df-8236-494b-b8b9-71481c94c0d9
# ╟─def60ead-a40b-4376-82cd-a77455f6b942
# ╠═67486053-5971-406b-8680-0d80803d797a
# ╠═d1cd6a46-dc5b-4188-a891-703b50bce186
# ╟─7f9e91b8-ee23-4d73-bfe5-c58a29b77abe
# ╟─942f314e-927b-4371-8c83-83801c860b4d
# ╟─2eb005a3-f5b2-4216-b56f-e25157b8c33c
# ╠═6016a736-11da-4451-aa82-cc3045e782db
# ╟─6f75889b-1c7f-4261-bf27-7c991ee9e414
# ╟─74290eff-781b-44c9-8a90-96bffbe040df
# ╠═34bad558-e70f-4d46-a9ab-7acc6c89db7a
# ╟─9049eeca-0db8-41d2-93ee-e0b4e445c9fd
# ╠═bf50534d-1c9a-439a-9922-262f64b83c1d
# ╟─06e8320c-ddd9-4d13-bca3-10fb5c3fb7ad
# ╟─ca1f0910-d417-41bc-ae2d-eebec7f3e1e9
# ╠═fc64996f-54ba-4c7a-8cfb-21133cec1fbe
# ╟─1f1e9c9b-e5fa-41b1-852f-cadad703ee4b
# ╠═e3dbe2f5-7e97-45b7-9b75-1acaf8f1031b
# ╟─39414e5e-1256-4497-9738-e2ecdff62d9d
# ╟─06dee467-1f54-48e1-908d-8e4c9028a748
# ╠═aa7b8b58-f959-47de-84d7-8c9cf3ad96be
# ╟─267532b6-ba74-4e74-992a-6cabb03949a0
# ╟─efb34c1a-5505-49f1-aa7f-24f6fd1fc01d
# ╠═2183e956-1b47-4e97-b957-c5df0541ff7b
# ╟─3996d09f-c115-4097-b994-6f3f573912fc
# ╟─6fb68c61-1ef0-4efc-bcbc-dd9d219c3ebb
# ╟─a9163072-cad9-4b0b-b154-d315c6b68de4
# ╟─fe2c7c2b-63e3-4cbf-b432-b028ec599292
# ╟─e2890775-2e29-4244-adac-c37f8f2a8a8e
# ╠═9ce43ec1-78f5-414f-9796-ab3159be7985
# ╟─a2466cba-65ea-41a4-b222-c397614453b2
# ╟─73444608-d0de-440d-8be3-5a02dadcadc7
# ╠═95f3bbad-1309-40c5-9da2-e1255a325d8b
# ╠═d462dc39-f90b-4429-b6b5-7ded05fa3432
# ╟─a1671960-9b0b-47f2-8d3a-74d67a122ce0
# ╠═8ea2e159-ef17-4ddd-b5a7-5f6c8d67238a
# ╟─b0773555-44ac-4b06-a410-d25ee1f42399
# ╟─df8f0019-84b9-4309-ab16-4a909fa94e88
# ╠═d4c08329-baa4-438a-9555-ee0ea5443e44
# ╟─954f9925-3346-49b6-8446-264e617efabc
# ╠═2909ff18-e98b-4149-843f-1c4709cfbb37
# ╟─ed7cb5c1-528a-4356-80dd-b337107eaf1f
# ╟─1e12834c-4b29-41db-ab1f-d93db62c8341
# ╠═09877b51-c34c-4351-ab8d-0fbee76d5a1b
# ╟─ba59678b-1606-4132-9f19-0dae1e660195
# ╠═dfebe8a3-fbe5-4381-bce5-1cd403a7b365
# ╟─b53798f9-24c2-4def-ab6f-447a5d809865
# ╠═90a42425-9f1b-464a-9b10-d0a25cc6717c
# ╟─24b1008e-f038-4d3d-a7f0-43d4488387f4
# ╟─3316bd55-0f83-48d1-8512-f9192953d716
# ╟─5505fc32-1e46-4256-831c-d1b94d1e946c
# ╟─94a9846b-ff01-487d-aeac-ddd4ab81610c
# ╟─d2e5bde1-8c65-494f-8944-b16dec6ab193
# ╠═655c8b62-e180-41e6-a3b5-7317cdc76f73
# ╟─6f2399db-24e7-4586-a93e-bdb38279470f
# ╟─ea518070-cc7d-4a33-b9fd-082f7f1aeca1
# ╠═82a245fa-6e02-41f4-bba4-769863a896db
# ╟─cc67a1d7-f195-4deb-9b39-22e06a75283d
# ╠═dda5a074-3f7b-46dd-bc4b-cb05117ec425
# ╟─29fb4486-5605-438f-9b1a-a24a19b20c5e
# ╠═f52b2954-6edb-48cc-8dc9-94a4ef613012
# ╟─a96569a4-4ab2-4681-ab4f-fae744a0a671
# ╟─5b237453-472f-414e-95e0-f44e980ea93a
# ╠═ef43aef6-80ec-4976-91e6-84a74d29a83e
# ╠═f75ad936-8c06-4e00-92d7-1f86532c0072
# ╠═2cf7fd33-2fda-4311-b699-c8441181b292
# ╠═54ef467e-8cc2-436e-bddc-37c57b6e6980
# ╠═b176823e-b8b5-413d-87b1-90d7efa0e377
# ╟─c75744a0-3c3f-4042-a796-6cbd9ec11195
# ╠═879e4355-d4f0-492c-9ac2-7359c794a9fd
# ╠═b8ff692c-2ccb-455c-87c2-590177def1b4
# ╠═fa6a31d1-f1b2-489b-9fef-4f92b5542c80
# ╠═0103f249-52e2-4403-b62e-b3ad21536d37
# ╠═e06d0925-585f-42e9-9dea-49044625d3d8
# ╠═aa8c6266-0338-44a9-b14b-088f98b04db3
# ╠═9b96ba97-1fe6-4d08-aa69-07a04219686b
# ╠═c6218e7c-1999-4e90-8868-a81717837fb2
# ╠═186680f8-9b47-4a93-9fa0-81c0c2ea894b
# ╠═fb5e7b56-66e5-45ef-9ad7-b50dd997137d
# ╠═94629920-3a9a-4d50-973f-cd18d3ea50f0
# ╠═38b63169-2712-48a6-b8b5-66429565cc87
# ╟─d8983a9d-1880-4dc4-9c17-23281767e0c2
# ╠═5e7bda42-0266-4498-906d-9aca8b6c4bf3
# ╠═805d2824-86cc-45bd-88b0-e6e14d9fde48
# ╠═3f4bffbb-51f4-4446-9b9c-cd3f99edfefa
# ╠═04c7209e-c4f8-454b-a883-cb2c5fac5203
# ╠═7fa55e99-5c0c-465d-a879-bd844e516131
# ╟─ddee1cf7-7977-407d-a004-08b52f6ff8c8
# ╠═485c046d-8329-4541-bd9d-eb180c01bde6
# ╠═b9318fcf-117e-438e-8bb4-985a9372e2d8
# ╠═3731fa3f-efb3-43d1-900f-8582cd89d526
# ╟─ba0545e7-c6df-42e2-a9cd-4ecd490d13e8
# ╠═510f07d9-62e0-40a7-b974-e2ae9bad7f73
# ╠═905793d5-93c5-4d86-9a88-33d6d806d88a
# ╠═10b73d22-cf2a-479c-84b6-6a63a694f398
# ╟─318a6643-5377-4152-8468-51dae1b78144
# ╠═4748a526-8d2e-43a6-8f30-82abf238d624
# ╠═bddafce9-30e4-4708-96ae-938bff9edfe7
# ╠═ebb0c754-12f1-4f80-a5f6-98a61b915fa6
# ╠═2d730c2f-7320-4879-b6a2-bee8c7c9b338
# ╠═a5fe54fb-4f92-4a35-8038-8d36a4aa065c
# ╠═86e58e39-186e-470f-832a-32cd86717daa
# ╠═0a06b151-461a-470b-927b-851c64d826bf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
