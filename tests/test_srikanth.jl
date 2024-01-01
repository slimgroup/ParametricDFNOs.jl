using Pkg
Pkg.activate("./")

using ParametricOperators
using LinearAlgebra
using Zygote

n = 2
T = Float32

U = ParMatrix(T, 2*n,2*n)
V = ParMatrix(T, n,n)
W = ParMatrix(T, n,n)

X = rand(T, 2*n,2*n)
Id = ParIdentity(T, 2*n)

theta_U = init(U)
theta_V = init(V)
theta_W = init(W)

 @time grads_L = gradient(theta_U -> norm((Id ⊗ (U(theta_U)*(V(theta_V)⊗W(theta_W))))*vec(X)), theta_U)[1]

#  My Philosophy is to work at the intersection of

#  1. Things that are impactful
#  2. Things that I love to work on
#  3. Things that I can make a living out of

# Quite frankly until I came across this posting I did not think I would find something that satisfied all three criterias and was convinced that I would need to start something of my own.

# I am enthralled to have the opportunity to apply to spellbrush!

# My experience includes optimizing LLMs at AmeriSave Mortgage, conducting large-scale model analysis, and engaging in complex computational methods at the Seismic Laboratory. 
# My expertise in training and optimizing large models, alongside a passion for AI adaptation and specialization, aligns well with Spellbrush's innovative endeavors in AI. 

# My lab also works on diffusion based models in the context of carbon sequestration monitoring and medical imaging. I would love to have the chance to talk more about my qualifications to join spellbrush.
# My courses such as deep learning, ML Theory and applied projects using tools such as jax, julia, pytorch and building my own framework has given me sufficient background in conducting research.

# Sample of my work that I presented this year: https://slim.gatech.edu/content/large-scale-parametric-pde-approximations-model-parallel-fourier-neural-operators

# PS. Makima was a close second


# Your original text is well-structured and clear, with a personal touch reflecting your enthusiasm and qualifications. Here's the refined version with corrected spelling and grammar, along with any minor changes for clarity:

# My philosophy is to work at the intersection of three key areas:

# Things that are impactful,
# Things that I love to work on, and
# Things that I can make a living out of.
# Quite frankly, until I came across this posting, I did not think I would find something that satisfied all three criteria and was convinced I would need to start my own venture.

# I am thrilled to have the opportunity to apply to Spellbrush!

# My experience includes optimizing Large Language Models (LLMs) at AmeriSave Mortgage, conducting large-scale model analysis, and engaging in complex computational methods at the Seismic Laboratory. My expertise in training and optimizing large models, alongside a passion for AI adaptation and specialization, aligns well with Spellbrush's innovative endeavors in AI.

# My lab also works on diffusion-based models in the context of carbon sequestration monitoring and medical imaging. I would love the chance to discuss my qualifications further and join Spellbrush. My courses in deep learning, ML theory, and applied projects using tools such as JAX, Julia, PyTorch, and building my own framework have provided me with a solid background in conducting research.

# Here's a sample of my work that I presented this year: https://slim.gatech.edu/content/large-scale-parametric-pde-approximations-model-parallel-fourier-neural-operators

# P.S. Makima was a close second.






