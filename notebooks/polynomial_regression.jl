### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ ef43aef6-80ec-4976-91e6-84a74d29a83e
using PlutoUI

# ╔═╡ 2c05ac8e-7e0b-4d28-ad45-24e9c21aa882
md"""
# Introduction

In this example, we introduce regression models as an example of the supervised learning paradigm.
"""

# ╔═╡ 18198312-54c3-4b4f-b865-3a6a775ce483
md"""
# The data

Here, our data is XXX.
We treat XXX as our features and XXX as our observations.
[plot data]
"""

# ╔═╡ cd49e0a5-4120-481a-965e-72e7bdaf867c
md"""
# Regression models

Regression has its own terminology we will avoid here.
We write instead $f(x)$, where $f(x)$ is an unknown function we want to approximate.

# Loss function

We then choose as our loss the L2 norm: $L(x) = -\lVert y - f(x) \rVert^2$

$(HTML(\"""
<details>
</details>
\"""))
"""

# ╔═╡ ae5d8669-f4c4-4b55-9af9-8488e43bcb6c
md"""
# Linear regression

For a linear regression model, we choose $f(x) = \alpha x + \beta$.
This model assumes that the function can be well approximated by a straight line.
[fit]
"""

# ╔═╡ 74290eff-781b-44c9-8a90-96bffbe040df
md"""
# Model expansion

Fitting the model gave us the best linear fit, which is obviously not great.
To get a better fit, we need to choose a different form for $f(x)$, for which we have 2 strategies:
1. Use what we know about the process approximated by $f$ to write down a more reasonable expression
2. Use a form for $f$ that is highly expressive.

Approach (1) is more useful when we know a lot about the mechanism and/or we want to know more about any properties of the mechanism.
Approach (2) is more useful when we don't know much about the mechanism and don't necessarily need to; we just care about good approximations.
For this example, we take approach (2).
"""

# ╔═╡ ca1f0910-d417-41bc-ae2d-eebec7f3e1e9
md"""
# Polynomial regression

A useful expression for $f$ is

$$f(x) = \alpha_0 + \alpha_1 x + \alpha_2 x^2 + \alpha_3 x^3 + \ldots \alpha_n x^n = \sum_{i=0}^n \alpha_i x^i$$

Functions of the form of $f$ are called _polynomials_, and any smooth function can be exactly computed with infinite terms (i.e. $n \to \infty$) or approximated with finite terms (by picking some manageable $n$).
Linear regression is the special case $n=1$.
"""

# ╔═╡ b0cdc9d6-738a-4583-b821-052ada846d39
# fit polynomial with slider to change n from, say, 0 to 100

# ╔═╡ 1f1e9c9b-e5fa-41b1-852f-cadad703ee4b
md"""
# Model complexity

We see that by increasing $n$, we fit the data better and better.
But, at some point it seems we're just playing connect-the-dots between the data-points.
[talk about model complexity, ask questions about whether such a detailed model is reasonable and discuss when it is not.]
"""

# ╔═╡ 39414e5e-1256-4497-9738-e2ecdff62d9d
md"""
# Cross-validation

A reasonable way to choose $n$ is by evaluating predictive performance on left-out data.
[Introduce cross-validation. Compute 5-fold CV. Plot MSE of test data as a function of $n$.]
"""

# ╔═╡ 73444608-d0de-440d-8be3-5a02dadcadc7
md"""
# Handling uncertainty

All along we've been showing the single best fitting function, but data is noisy and incomplete, so we can't be 100% certain about what that best fitting function is.
Regression models are statistical models, and specifically, we can write them as Bayesian models.
In a "Bayesian" model (or, just a "probabilistic model"), uncertainties are quantified with probabilities.
The polynomial regression model we have been using is written
"""

# ╔═╡ 3fc67210-6df1-481b-8aa4-48865f9f46b5
md"""
What we have been doing so far is computing the maximum-likelihood (a posteriori) estimate of this regression model, i.e. the set of parameters that distribution of parameters given data has its peak [show visualization].
To quantify the uncertainty, we can instead visualize the distribution of parameters.
Each set of parameters defines a curve, so we can visualize this distribution by visualizing random curves:
"""

# ╔═╡ e0ec835f-e9dd-4660-93bc-c26d0ba732e5
md"""
Alternatively, we can get draw, say, 90% intervals around the single best fit to get a lower-bound on the uncertainty.
"""

# ╔═╡ c7912586-f4f6-450b-999f-936339898997
md"""
# Relationship to neural nets

[not certain if we want this]
"""

# ╔═╡ 3991be1c-01f3-416a-a841-18f025a97e24
md"""
# Regression for classification problems

Now, $f$ maps from continus features to continuous observations, but what about a binary classification problem, where $y$ can take on values of $0$ and $1$?
[show data]
"""

# ╔═╡ ba59678b-1606-4132-9f19-0dae1e660195
md"""
We can keep almost everything the same here, but what we need to do is add a nonlinearity to map from the real output of $f$ to the interval $[0, 1]$.
We can choose the sigmoidal (S-shaped) _logistic function_ to do this, and the result is logistic regression:
[formula]
"""

# ╔═╡ ed7cb5c1-528a-4356-80dd-b337107eaf1f
md"""
[show fit]
"""

# ╔═╡ f75ad936-8c06-4e00-92d7-1f86532c0072
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.37"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═2c05ac8e-7e0b-4d28-ad45-24e9c21aa882
# ╠═18198312-54c3-4b4f-b865-3a6a775ce483
# ╠═cd49e0a5-4120-481a-965e-72e7bdaf867c
# ╠═ae5d8669-f4c4-4b55-9af9-8488e43bcb6c
# ╠═74290eff-781b-44c9-8a90-96bffbe040df
# ╠═ca1f0910-d417-41bc-ae2d-eebec7f3e1e9
# ╠═b0cdc9d6-738a-4583-b821-052ada846d39
# ╠═1f1e9c9b-e5fa-41b1-852f-cadad703ee4b
# ╠═39414e5e-1256-4497-9738-e2ecdff62d9d
# ╠═73444608-d0de-440d-8be3-5a02dadcadc7
# ╠═3fc67210-6df1-481b-8aa4-48865f9f46b5
# ╠═e0ec835f-e9dd-4660-93bc-c26d0ba732e5
# ╠═c7912586-f4f6-450b-999f-936339898997
# ╠═3991be1c-01f3-416a-a841-18f025a97e24
# ╠═ba59678b-1606-4132-9f19-0dae1e660195
# ╠═ed7cb5c1-528a-4356-80dd-b337107eaf1f
# ╠═ef43aef6-80ec-4976-91e6-84a74d29a83e
# ╠═f75ad936-8c06-4e00-92d7-1f86532c0072
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
