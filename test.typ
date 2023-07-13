
#set page(
  paper: "a4"
)


= Image formation model


- Images in Fourie space:
$ cal(X) = {XX_i}_1^n = XX_1, ...,  XX_i, ...,XX_n in CC^l^2 "where" l = "boxsize" $

- Volume in Fourie space: 
$ VV in CC^l^3 $

- Projection operator:
$ HH_q_i in CC^(l^2 times l^3) $ where $q in  QQ := "SE(3)"$ describe Euler angles and translation. 

$ XX_i = HH_q_i  VV $

If we assume CTF effects are linear, it could be describe in $HH$.

= Probability model

- Some important formulae

$ a in A , b in B $
$ P(A) = sum_(a in A) P(a) $
$ P({a_i}_1^n) = P(cal(A)) = product_i^n P(a_i) $
$ P(a) &= sum_(b_i in B) P(a,b) \
       &= integral_(B) P(a,b) dif b $
$ P(a|b) = P(a,b) / P(b) = P(a,b) / (sum_A P(a,b) ) $


We then seek the maximum a posteriori (MAP) estimate of $VV$ by maximizing the following

$ P(VV|cal(X)) &= P(VV,cal(X)) / P(cal(X)) = (P(cal(X)|VV)P(VV)) / P(cal(X)) \ $

and regularized log-likelihood form is following

$ cal(l)(VV|cal(X)) = log P(VV|cal(X)) &= log( P(cal(X)|VV)P(VV) / P(cal(X))) \
 &=  log P(cal(X)|VV) + log P(VV) / P(cal(X))
$

As this is not a computable form about $P(cal(X)|VV)$ , we use EM-algorithm to explicitly include the projection model.

$
log P(cal(X)|VV) &= log sum_(q in Q) P(cal(X),q|VV) \
&= log sum_(q in Q) p(q) P(cal(X),q|VV) / p(q)\
"(Jensen's inequality)" &>=  sum_(q in Q) p(q)  log P(cal(X),q|VV) / p(q) \
cal(L)(cal(X)|VV) :&= sum_(q in Q) p(q)  log P(cal(X),q|VV) / p(q)\
log P(cal(X)|VV) - cal(L)(cal(X)|VV) &= sum_(q in Q) p(q){ log P(cal(X)|VV)  - log P(cal(X),q|VV) / p(q)} \
&= sum_(q in Q) p(q){ log p(q) P(cal(X)|VV) / P(cal(X),q|VV)  } \
&= sum_(q in Q) p(q){ log p(q)  / P(q|cal(X),VV)} >= 0 \
"KL"(p|P) :&= sum_(q in Q) p(q){ log p(q)  / P(q|cal(X),VV)} >= 0 \
$

Then in E-step, we calculate following

$
hat(P)_j = op("argmax",limits: #true)_p(q)  P(cal(X)|VV_j) 
$

If $ hat(P)_j = Gamma(q) = P(q|cal(X),VV_j)$, $"KL"(p|P) = 0 => log P(cal(X)|VV_j) = cal(L)(cal(X)|VV_j) $, and $log P(cal(X)|VV_j)$ don't depends on $p(q)$, its value is constant.

$
Gamma(q) &= P(cal(X), q|VV) / P(cal(X)| VV)  = P(cal(X), q|VV) / (integral_Q P(cal(X), q|VV)) \
&= (P(cal(X)| q,VV) P(q|VV)) / (integral_Q P(cal(X)| q,VV) P(q|VV)) \ 
$



We can model this form by following

$ 
P(XX_i|q,VV) = C / sqrt(det(sigma^2)) exp(-1/2 norm(sigma^(-1)(XX_i-HH_q^i VV))^2 )\
"where" sigma := "diag"({mono(sigma)_i}_1^(l^2) ) in RR_+^(l^2 times l^2)   
$ 

$
P(VV) = C / sqrt(det(tau^2)) exp(- 1/2 norm( tau^(-1) VV)^2)\
"where" tau := "diag"({mono(tau)_i}_1^(l^3) ) in RR_+^(l^3 times l^3)   
$

The noise parameter $sigma$ are often modeled as  the resolution-dependent variance.

As $P(q|VV) approx P(q) $ is a probability for angle and movement, it is intuitively a uniform distribution, but some assumptions may be made to speed up the calculation (i.e. local search, branch and bound).

The prior $P(VV)$ expresses how likely that model is given the prior information. Since the protein is made up of atoms, the amplitude in reciprocal space is the sum of these numbers. The phase represents the position of each atom and, assuming that the positions of the atoms in the protein are random, the phase of the whole protein can be regarded as equivalent to a random walk in a two-dimensional plane and can therefore be modelled by a Gaussian distribution with zero mean and resolution-dependent variance $tau$.

So we summarize E-step. 

$
hat(cal(l))(VV|cal(X)) &= sum_(q in Q)   Gamma(q)_j log P(cal(X),q|VV) / (Gamma(q)_j) + log P(VV) / P(cal(X)) \ 
&= sum_(q in Q)   Gamma(q)_j log ( P(cal(X)|q,VV) P(q) )/ (Gamma(q)_j) + log P(VV) / P(cal(X))
$

Next M-step is following

$
hat(VV)_(j+1) = op("argmax",limits: #true)_VV  hat(cal(l))(VV|cal(X))
$

and solve $nabla_VV hat(cal(l))(VV|cal(X)) = 0$ to find $VV$.

$
nabla_VV hat(cal(l))(VV|cal(X)) &= diff / (diff VV) hat(cal(l))(VV|cal(X)) \
&= diff / (diff VV) {sum_(q in Q)   Gamma(q)_j log  P(cal(X)|q,VV)   + log P(VV) } \
&= diff / (diff VV) { -  sum_i^n sum_(q in Q)  Gamma_i (q)_j norm(sigma^(-1)(XX_i-HH_q^i VV))^2    - norm( tau^(-1) VV)^2 } \
&=  sum_i^n sum_(q in Q)  Gamma_i (q)_j HH_q^(*,i) sigma^(-2)(XX_i-HH_q^i VV)    - tau^(-2) VV \
$

Then, in the Maximization step we solve for the parameters of a $VV$, which yields the closed-form solution:

$
VV_(j+1) = ( sum_i^n sum_(q in Q)  Gamma_i (q)_j HH_q^(*,i) sigma^(-2)HH_q^i + tau^(-2))^(-1)( sum_i^n sum_(q in Q)  Gamma_i (q)_j HH_q^(*,i) sigma^(-2)XX_i  )
:= KK^(-1)  BB
$

The  $HH^* sigma ^(-2) HH$ in $KK$ means backprojected sigma. Note atleast we want keep 3 volume in VRAM. (reference, backprojection volume, backprojection weight = $KK$)

With same manner, we solve $sigma_(j+1), tau_(j+1)$.

$
diff / (diff sigma^2) hat(cal(l))(VV|cal(X))
&= diff / (diff sigma^2) sum_(q in Q)   Gamma (q)_j log  P(cal(X)|q,VV) \
&= diff / (diff sigma^2) { - sum_i^n sum_(q in Q)  Gamma_i (q)_j ( 1/2 norm(sigma^(-1)(XX_i-HH_q^i VV))^2  + log det(sigma^2) ) } \
&=  sum_i^n sum_(q in Q)  Gamma_i (q)_j ( 1/2 sigma^(-2) (XX_i-HH_q^i VV)(XX_i-HH_q^i VV)^* sigma^(-2) - sigma^(-2) ) \ 

&= sigma^(-2) sum_i^n sum_(q in Q)  Gamma_i (q)_j ( 1/2 sigma^(-2) (XX_i-HH_q^i VV)(XX_i-HH_q^i VV)^* - II ) \



\
diff / (diff tau^2) hat(cal(l))(VV|cal(X))
&= diff / (diff tau^2) log  P(VV) \
&= diff / (diff tau^2) { - ( 1/2 norm(tau^(-1)VV)^2  + log det(tau^2) ) } \
&= 1/2 tau^(-2)VV VV^* tau^(-2) -tau^(-2)\

sigma^(2)_(j+1) &= 1/(2n) (sum_i^n sum_(q in Q) Gamma_i (q)_j (XX_i-HH_q^i VV)(XX_i-HH_q^i VV)^* )
\
tau^(2)_(j+1) &= 1/2 VV VV^*
$

= Summary

$
P(XX_i|q,VV) = C / sqrt(det(sigma^2)) exp(-1/2 norm(sigma^(-1)(XX_i-HH_q^i VV))^2 )\
Gamma_i (q) = (P(XX_i| q,VV) P(q|VV)) / (integral_Q P(XX_i| q,VV) P(q|VV)) \
VV_(j+1) = ( sum_i^n sum_(q in Q)  Gamma_i (q)_j HH_q^(*,i) sigma^(-2)HH_q^i + tau^(-2))^(-1)( sum_i^n sum_(q in Q)  Gamma_i (q)_j HH_q^(*,i) sigma^(-2)XX_i  )\
sigma^(2)_(j+1) = 1/(2n) (sum_i^n sum_(q in Q) Gamma_i (q)_j (XX_i-HH_q^i VV)(XX_i-HH_q^i VV)^* )
\
tau^(2)_(j+1) = 1/2 VV VV^*
$