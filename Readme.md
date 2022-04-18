## Spheroidal dynamics
A python code that solves dynamical problems of interacting polar particles on spheroidal geometries. 

Non-interacting particles follow geodesics and interactions are modeled as repulsive contacts. 

<p align="center">
  <img src="animation/dyn_anim.gif" alt="animated" />
</p>

The spheroid has two major axes with length *a* and a minor axis with length *b*. Particles are described with three parameters 

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=r = \phi, \theta, \psi,">
</p>

where the first two are the pseudo-spherical coordinates of the particle on the spheroid and the latter is their polar angle. The pseudo-spherical coordinate system is a projection of the particle position on the spheroid to a sphere. The Cartesian vector pointing to a particle on the spheroid is then expressed as:


<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=r = L(\eta) \left( \sin \eta \cos \phi, \sin \eta \sin \phi, \cos \eta \right)^T">
</p>

with

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=r = \sin(\eta) = \frac{a \sin \theta}{\sqrt{a^2 \sin^2\theta+b^2\cos^2\theta}},">
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=r = \cos(\eta) = \frac{b \cos \theta}{\sqrt{a^2 \sin^2\theta+b^2\cos^2\theta}},">
</p>

and

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=r = L(\eta) = \frac{ab}{\sqrt{a^2 \cos^2\eta+b^2\sin^2\eta}}.">
</p>