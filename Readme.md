## Spheroidal dynamics
A python code that solves dynamical problems of interacting polar particles on spheroidal geometries. 

Non-interacting particles follow geodesics and interactions are modeled as repulsive contacts. 

![](animation/dynamics.mp4)

The spheroid has two major axes with length $a$ and a minor axis with length $b$. Particles are described with three parameters 

$$\phi, \theta, \psi,$$

where the first two are the pseudo-spherical coordinates of the particle on the spheroid and the latter is their polar angle. The pseudo-spherical coordinate system is a projection of the particle position on the spheroid to a sphere. The Cartesian vector pointing to a particle on the spheroid is then expressed as:

$$r = L(\eta) \left( \sin \eta \cos \phi, \sin \eta \sin \phi, \cos \eta \right)^T $$

with


$$\sin(\eta) = \frac{a \sin \theta}{\sqrt{a^2 \sin^2\theta+b^2\cos^2\theta}},$$

$$\cos(\eta) = \frac{b \cos \theta}{\sqrt{a^2 \sin^2\theta+b^2\cos^2\theta}},$$

and

$$L(\eta) = \frac{ab}{\sqrt{a^2 \cos^2\eta+b^2\sin^2\eta}}.$$