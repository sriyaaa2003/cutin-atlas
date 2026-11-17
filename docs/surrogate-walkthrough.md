# The finite element + graph network study, in plain words

This is the explanation to read before talking about the study. It covers what each piece is, why it
is there, and what the results do and do not show. The numbers are in `results/surrogate/` and in the
README section "Surrogate study".

## The question

The rest of cutin-atlas asks how few expensive simulation runs are enough to answer a question. This
study asks the same for a **finite element (FEM) simulation**: if a graph neural network learns from
a handful of FEM runs, how close does it get on plates it has never seen, and how many runs does it
need before it beats a simple formula?

## What a finite element simulation is (`fem.py`)

- A **plate** (a rectangle, with or without a round hole) is clamped on its left edge and pushed on
  its right edge.
- The plate is cut into small **triangles** (the *mesh*). Inside each triangle the strain is treated
  as constant, which makes the maths simple (these are called constant-strain triangles).
- Each triangle has a small **stiffness matrix** that says how much force it takes to move its three
  corners. Adding all of them together gives one big sparse matrix `K` for the whole plate.
- The **boundary conditions** say which corners are held (the left edge) and where the force is applied
  (the right edge). Then the solver solves `K u = f` for the movement `u` of every corner.
- From the movement it computes the **stress** in every triangle and combines it into one number, the
  *von Mises stress*, which engineers compare with a material limit.
- The hole is cut by deleting the triangles whose centre lies inside the circle, so its edge is a
  staircase. That is a simple choice and it is stated in the code.

## Why you can trust the solver (`tests/test_surrogate_fem.py`)

- **Patch test:** pull a plate evenly and the stress must be exactly even. It is, to about 1e-14.
- **Equilibrium:** the force the clamp pushes back with must equal the force applied. It does, to 1e-9.
- **Beam theory:** for a plate without a hole, the tip movement approaches the textbook beam formula as
  the mesh gets finer (91%, then 97%, then 99% of it). Triangles come out slightly too stiff, which is a
  known property of this element, and the error shrinks with the mesh.
- **Linearity:** doubling the load doubles every result.

## The graph (`graph.py`)

The mesh is already a graph: every mesh corner is a **node** and every triangle edge is an **edge**.
Each node gets ten numbers about itself (position, whether it is clamped, where the load acts, the plate
height, and where the hole is) plus the load direction. The targets are the movement and the von Mises
stress at every node.

Because the FEM is linear, a load twice as large gives results twice as large. So the network predicts
the result **per newton** and the load size is multiplied back afterwards. This removes something the
network would otherwise have to learn, and it is exact.

## The graph network (`gnn.py`)

- It starts by turning each node and each edge into a vector of 32 numbers.
- Then it does six rounds of **message passing**: every edge looks at its two end nodes, and every node
  adds up what its edges tell it. After six rounds a node has information from six steps around it.
- Finally a small decoder turns each node's vector into the three outputs.
- It is written in plain PyTorch, with no graph library.

Three models are compared:

| model | what it is | why it is there |
|---|---|---|
| `gnn` | the graph network learning the FEM result | the main idea |
| `gnn_res` | the same, but it only learns the **correction** to a beam formula | physics plus a learned residual, the same design as evprior |
| `mlp` | the same inputs, no message passing | shows what the graph structure adds |

The **beam formula** is the baseline everything has to beat. It is a textbook closed-form answer that
knows the load and the plate size but not the hole.

## What the study does (`study.py`)

It solves random plates (random height, hole size and place, load size and direction), trains each model
on the first 8, 16, 32, 64 and 128 of them with two random seeds, and scores each model on 40 other plates
that were never used for training or for choosing the design. The error is the relative size of the
difference between prediction and FEM, over the whole field.

## How to describe the result honestly

Read the actual numbers in the README. In general:

- A graph network that learns the whole answer from a few runs is worse than the simple beam formula; it
  needs many more runs to catch up. That is expected: the formula is nearly exact away from the hole.
- A graph network that learns only the correction to the formula beats the formula, because the
  correction is small and sits near the hole and the clamp.
- Removing message passing makes the model clearly worse, so the graph structure matters.
- At this plate size the FEM solver takes about 9 milliseconds, so the network is **not** faster. The
  study measures how many runs a surrogate needs, not speed. The payoff of a surrogate comes with
  simulations that take minutes or hours, which this small solver does not.

## What this is not

- It is a small two-dimensional study with a self-written solver, not an industrial CAE workflow.
- It has no crash simulation, no contact, no plasticity and no real geometry from a CAD model.
- It was built with AI assistance in this repository; be ready to say so and to explain every part.

## Questions you may be asked, and honest answers

- *Why constant-strain triangles?* They are the simplest element and exact for the patch test; the cost
  is that they are a bit stiff in bending, which the beam-theory test shows.
- *Why not a library like FEniCS or Abaqus?* To understand the method and keep the data fully checked;
  a small solver was enough for the question.
- *Why a graph network and not a CNN?* A mesh with a hole is not a regular grid; a graph follows the
  mesh directly and handles different shapes without resampling.
- *Does the network replace the solver?* Not here. It shows how much data a surrogate needs, and where a
  physics formula plus a learned correction is better than learning everything.
- *What would you do next?* Try the same on a three-dimensional or nonlinear problem where the solver
  is slow, and compare against a mesh network with more rounds of message passing.
