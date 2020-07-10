This tasks for numerical methods.   
Each program is launched by the command: python3 name_program.py   
But before, you must install python3 on your computer.  
All graphs are built depending on the turnkey solution and mine.  

# 1. gauss.py 
Gaussian elimination, also known as row reduction, is an algorithm in linear algebra for solving a system of linear equations. It is usually understood as a sequence of operations performed on the corresponding matrix of coefficients. This method can also be used to find the rank of a matrix, to calculate the determinant of a matrix, and to calculate the inverse of an invertible square matrix.
Gauss 1st example  
```
5
[0.10640311936426983, 0.12954233612776672, -0.041442969807068297, 0.30552155185092117, 0.086705317420873104]
[ 0.11688665  0.14450197 -0.10031595  0.25987603  0.08670532]
```
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/gauss1.png)  
Gauss 2nd example  
```
50
[0.022533836529470185, 0.0047157979827202731, 0.00057519545038903013, 0.0033187737745425426, 0.0085328725347283418, 0.025029029729529268, 0.018118304238861443, 0.024771963804080956, 0.024048422948933952, 0.021759911164609052, 0.0036674679405434047, 0.013477040989659584, 0.016194582187028662, 0.0048462753809155575, -0.0040977332047903289, -0.003962822268692644, -0.00508413386151232, 0.021471193727721417, 0.0027574833645547954, 0.018467204560861272, 0.013362784731847977, -0.0015721338141367444, 0.019959975050867357, 0.0053203583848649341, -0.0017279714444729153, 0.0061732563514212222, -0.0050303947690897698, 0.0034077718258709029, 0.015206829158895508, 1.1988058901920292e-05, 0.015182135465529559, -0.00032644442998167126, 0.0034698141018331642, -0.0026399302260777698, 0.0149896805346114, 0.016206730164148531, 0.014343527686301288, 0.011005764091961813, -0.0045231289854070333, -0.00082952880372293863, 0.018370066765954859, 0.0060173985151882325, -0.0036620538280801732, 0.00077507365862434824, -0.0034420205013817743, 0.0050792828086265078, 0.0047299738870204147, 0.01085008558129328, 0.0035312488648299666, 0.0063005166314225522]
[  3.11104919e-02   1.02419856e-03   1.15499051e-03   2.36755869e-04
   1.93367782e-02   4.34414192e-01   1.39859726e-03   2.25638346e-03
  -1.57642467e-03   6.82615344e-02  -2.20385337e-01  -3.22590955e-03
  -3.39407705e-03  -3.82982831e-05   3.45137445e-05  -6.37904020e-04
  -1.95348079e-03  -4.83243338e-04   6.64317567e-03   8.13296937e-04
  -6.37760351e-05   3.19932047e-03   6.86688186e-04   2.54056422e-04
   1.73041737e-03   1.68704004e-04  -4.76457046e-04   4.09124476e-03
   4.18288479e-04   1.20693285e-02   7.38044136e-04   7.00979144e-04
   1.82928478e-03  -4.35068979e-02   1.23809920e-03  -8.38913184e-04
  -1.14961829e-03   1.44301588e-04  -9.25886157e-05  -2.76459268e-03
   5.09203075e-03  -6.19129087e-04  -1.31119363e-05   5.60740334e-05
   2.06525476e-04   3.54261457e-04   2.28657681e-03  -4.89867679e-02
  -6.13574054e-01   6.30051663e-03]
```  

![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/gauss2.png)  
# 2. Progonka.py  
The sweep method is a special case of the Gauss method and is used to solve systems of linear equations of the form Ax = B,where A is a three-diagonal matrix. A three-diagonal matrix is ​​a matrix of the form where zeros are in all other places except the main diagonal and two adjacent to it.
The sweep method consists of two stages: direct sweep and reverse sweep. At the first stage, running coefficients are determined, and at the second, unknown x are found.  
Progonka 1st example  
```
5
[0.29346500324956881, 0.23312392498882306, -0.060120135580518451, 0.086870544948763409, -0.016842731964425102]
[ 0.28602979  0.23925317 -0.02400086  0.05375295 -0.00076072]

```
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/progon1.png)    
Progonka 2nd example  
```
50
[0.21199902128662651, -0.062012904646452968, 0.19463848411845339, 0.12620121410046439, 0.022423051217254047, 0.60837433457373358, -0.40204206123485964, 0.61253475659992573, -0.043303349576955259, -0.026761210447170183, 0.20187947725042235, 0.35950353039848687, 0.54764721553787865, -0.33362646317167188, 0.92798074638370087, -0.15058557459070887, 0.19980163463805795, 0.48259624301173781, -1.1691272655816185, 2.0562504804197159, -0.8030334809120776, 0.6260877919552108, -0.12751543378694796, 0.52442424671085397, -0.14327436635202107, 0.76166832042586419, -0.30708689171193387, 0.4588744802366137, -0.1496020258204929, 0.64580124789712001, -0.44133109275430649, 1.1328368910451896, 0.025813750461639511, 0.06499293253030658, 0.59092831456291961, 0.28184911221268349, 0.046056181346331937, 0.29425052953600633, 0.68776523945455093, 0.036816202919562349, 0.24110792633339428, -0.23665503032544546, 0.70476335170907733, -0.320352024106205, 0.72720339877548257, -0.42962566950378284, 0.38703903405951001, 0.32118504599523667, 0.13878927538421976, 0.037917640432784276]
[ 0.20883787 -0.02927614  0.19424238  0.10844138  0.04564787  0.55707283
 -0.46195672  0.70791567 -0.34814598 -0.06327504  0.33075934  0.4799549
  0.13371696 -0.23377997  0.66556232  0.14109345  0.10402044  0.27785801
 -0.28204283  1.32056468  0.4609795   0.03002222  0.31904101 -0.04362255
  0.21440003  0.63252158 -0.12614472  0.3812343  -0.40331606  0.66076667
 -0.43619362  1.2220014  -0.02225422  0.28831416  0.24677267  0.51659135
 -0.49630825  0.84832862 -0.02749995  0.35196346  0.31791154 -0.2318759
  0.69601952 -0.22650424  0.55702025 -0.12638944 -0.1723939   0.58837524
 -0.92652315  0.376901  ]
```
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/progon2.png)  
# 3.Seidel.py  
Метод Зейделя (иногда называемый методом Гаусса-Зейделя) является модификацией метода простой итерации, заключающейся в том, что при вычислении очередного приближения x(k+1) (см. формулы (1.13),(1.14)) его уже полученные компоненты x1(k+1), ...,xi - 1(k+1) сразу же используются для вычисления xi(k+1).
В координатной форме записи метод Зейделя имеет вид:
x1(k+1) = c11x1(k) + c12x2(k) + ... + c1n-1xn-1(k) + c1nxn(k) + d1
x2(k+1) = c21x1(k+1) + c22x2(k) + ... + c2n-1xn-1(k) + c2nxn(k) + d2
...
xn(k+1) = cn1x1(k+1) + cn2x2(k+1) + ... + cnn-1xn-1(k+1) + cnnxn(k) + dn
где x(0) - некоторое начальное приближение к решению.
Таким образом i-тая компонента (k+1)-го приближения вычисляется по формуле
xi(k+1) = ∑ j=1i-1 cijxj(k+1) + ∑ nj=i cijxj(k) + di , i = 1, ..., n 	(1.20)
Условие окончания итерационного процесса Зейделя при достижении точности ε в упрощенной форме имеет вид:
|| x (k+1) - x (k) || ≤ ε.  
Seidel 1st example
```
5
[0.0082342450270267292, 0.20217673357085111, 0.18485920056979654, 0.046309631137482918, -0.050266872405383883]
[-0.05174796  0.14811635  0.20290369  0.06149709 -0.04543242]

```
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/seidel1.png)  
Seidel 2nd example  
```
50
[0.0021207461415912714, 0.010825139974867799, -0.0084535788358896542, 0.029151529988861766, 0.023150206262280294, -0.0017346452632047186, 0.021862184042752312, 0.023563121137468227, 0.00061325749304545285, -0.00090361171957454287, 0.02142075617108263, 0.0020475579495410467, 0.016133345651094651, 0.0010568216135466715, -0.0052310740701064879, 0.018294317944643732, -0.0017828777417061067, 0.014570930006433737, 0.023425652921635273, 0.017701604824877523, 0.017254377402574125, 0.02274167213147503, 0.011410639157673099, 0.0085420250660292116, 0.011472539310604784, 0.006659590622231139, 0.0067549818968256903, 0.010264796457474036, 0.02069701254208969, 0.014774623362320058, -0.0040683844485738989, 3.8782724524167653e-05, 0.0041267880527936888, 0.00076282932681624607, 0.0049393399316708161, 0.0081756451645280462, 0.010191588447951008, 0.010607357262555534, 0.017847096019800224, 0.0082242937124368549, 0.0032043836813814143, 0.0061582707851174923, 0.014766091560747877, 0.0022149591334141611, -0.00072612029848510395, 0.0057790865940719523, 0.0021651794953386088, 0.00046898840499798934, 0.0076163937427385373, -0.0037991499823027217]
[ 0.00372327  0.0122241  -0.00739039  0.03041338  0.02387327 -0.00085177
  0.0225522   0.0240329   0.00111643 -0.0005413   0.02153862  0.00224768
  0.01613643  0.00111708 -0.00532795  0.01816263 -0.0019147   0.01443984
  0.02321241  0.01755848  0.0170968   0.02253674  0.0112488   0.00834205
  0.01121374  0.00648822  0.00651027  0.01005456  0.02052001  0.01452832
 -0.00425458 -0.00017821  0.0039112   0.00059855  0.00479654  0.00792221
  0.0100471   0.01040938  0.01772024  0.00809011  0.00309065  0.00604651
  0.01465711  0.0021277  -0.00079656  0.00569997  0.00205977  0.00037992
  0.0075706  -0.00383202]

```

![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/seidel2.png)  
# 4. Jakoby.py  
The Jacoby method is one of the simplest methods to bring the matrix system to a form convenient for iteration: from the 1st equation of the matrix we express the unknown x1, from the 2nd we express the unknown x2
etc.The result is matrix B,in which on the main diagonal there are zero elements, and all the rest are calculated by the formula:
bij = −aij / aii, i, j = 1, 2 ..., n
Elements (components) of vector d
calculated by the following formula:
di = bi / aii, i = 1, 2, ..., n
The calculation formula of the simple iteration method:
x (n + 1) = Bx (x) + d
Matrix Record (coordinate):
xi (n + 1) = bi1xn1 + bi2x (n) 2 + ... + b
Ending criterion in the Jacobi method:
|| x (n + 1) −x (n) || <ε1
, where ε1 = 1− || B |||| B || ε
If B <1/2,then we can apply a simpler criterion for ending iterations:
|| x (n + 1) −x (n) || <ε  
Jakoby 1st example  
```
5
[0.0024060697429644493, 0.052217362344140997, 0.077204869988588304, 0.021654434411729276, 0.13025822281718594]
[-0.01737019  0.01060054  0.07118036  0.00398417  0.12004975]
```

![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/yakobi1.png)  
Jakoby 2nd example  
```
50
[0.013484694547678677, 0.015598083228154185, 0.014005338952023473, 0.02208695126973614, 0.0091592457239377395, 0.027963898279794754, 0.028144315471463184, 0.010416790072962791, -0.0044591391531552897, 0.0021937338372900084, 0.015922416486945375, -0.00093980492174455441, 0.0082681780854096075, 0.025236183467782211, 0.018064846607595948, 1.5297491585896853e-05, 0.021969077556432601, 0.012051169706450354, 0.022979173377797014, 0.020793540149865605, 0.016676458609771835, 0.0064382041864236477, 0.023400578222358106, 0.010558487975484979, 0.0063062895355343309, -0.0010676348750324513, 0.021398417598994249, 0.013952156668533494, -0.0024716398309169662, 0.0004893691103775126, 0.0088810344289261822, 0.0031560405386608075, 0.011759648973943814, 0.018946897873994343, 0.0093833222668972347, 0.015409256972226134, -0.0017198344087953114, 0.00046148575345748795, 0.023344407571892371, 0.01230422205127394, 0.0153721948648803, 0.0094771370824526906, 0.012678235338734894, 0.0069046911316427376, 0.018040621805430004, 0.0061189952359881139, 0.0009662809273301348, 0.0040384702031255083, 0.0037287938461700969, 0.0090291819343925241]
[ 0.00969247  0.01198778  0.01020591  0.01848139  0.00562357  0.02454969
  0.02472579  0.0072322  -0.00772387 -0.00094705  0.01274988 -0.0040128
  0.00533836  0.02234172  0.01515597 -0.0028037   0.01909136  0.00929551
  0.0201737   0.01814861  0.01400004  0.00364432  0.0207755   0.00804909
  0.00373848 -0.00367951  0.01890279  0.011497   -0.00496841 -0.00190976
  0.00649696  0.00069112  0.00938661  0.0165967   0.00714118  0.01315571
 -0.00392064 -0.00166412  0.02109923  0.01007147  0.01326028  0.00741204
  0.01056885  0.00482559  0.01602723  0.00409379 -0.00107826  0.00211351
  0.00186177  0.00709646]


```

![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/yakobi2.png)    
# 5. interpol_splain.py  
A spline is a function whose domain is divided into a finite number of segments, on each of which it coincides with some algebraic polynomial. The maximum degree of polynomials used is called the spline degree.  
1st Example
```
5
x: [ 4.14757698  0.2191087   1.10029986  2.49146852  3.64603921] 
y: [ 2.33114358  3.5038727   4.59877739  1.7693087   3.72638994] 
z: [ 4.91623522  1.14190025  3.86228083  3.35435711]
[ 2.10168319  4.65046703 -1.01875763  3.23196757]


```

![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/splain1.png)  
2nd example  
```
20
x: [ 15.19013615   4.23954671  14.08911985  15.53765455   4.12198488
  18.15744215   2.42358296  15.52185978   9.7244545   13.14641295
   6.21037381   1.54835963  15.9127913   15.60880921   7.29293244
   2.82686959  14.1594513    9.6815579   19.12314617  14.90299798] 
   y: [  2.90427748  19.56054067   2.22436496  18.27082984  17.32652559
   5.28510058  13.18126964   1.02578844   4.1741457   13.06524988
   7.48313675   9.48504441  12.81106155   4.19817683   4.43558497
  11.92342829  17.36999569  12.38511048   6.95812947   6.97275802] 
  z: [ 11.82769165   0.24593277   8.31941256  18.31119792   2.40469022
   6.89470841  11.80064847  14.07714421   3.34336133   2.82963474
  14.7435358   12.62393824  11.47794201  16.49985688   8.0138482
   2.22914668  13.20678863   6.20646825  11.62288028]
[   8.01868349   26.58967706  -61.69084909   18.50025736   18.79984239
   10.93739788    4.47915217    1.81036025  -12.40553086    4.76232339
    3.81892603   12.04954297 -112.84386007    4.17273851    3.22689076
   11.63615587   16.30947154   14.38257584    6.98412808]


```

![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/splain2.png)      
# 6. Interpol_lagr.py  
The Lagrange interpolation polynomial is a polynomial of minimal degree that takes these values ​​at the specified set of points. For n + 1 pairs of numbers (x0, y0), (x1, y1), ..., (xn, yn), where all xj are different, there exists a unique polynomial L (x) of degree at most n for which L (xj) = YJ.  
Lagrange1 example  
```
5
x: [ 0.50579171  0.97994765  3.42736679  3.4536123   3.19083841] y: [ 3.22711216  3.30251968  0.6015459   3.2058259   2.32332944] z: [ 3.58090676  1.3189894   0.28411579  2.95693035]
26.1323194201

```
![Image alt](https://github.comMcken09/Numerical-methods1/raw/master/pictures/lagr1.png)  
Lagrange2 example  
```
50
x: [ 40.38797403  46.37303413   8.08387368  10.29454408  29.20114856
   5.30702447  16.20223711  46.89561744  22.91536016  35.43848385
  46.97289565  12.37035787  36.79026147  46.73636754  34.59828412
  43.3838627   26.57794332  22.14147042  41.31563268  10.44291072
  42.43900006  22.40848151  40.2901286   21.28302344  31.45575774
   0.40988219  13.43397778  25.03746401   1.58012549   9.85318114
  19.55804692  35.90598252  19.81937947  47.60940249  19.66686966
  42.26291862  44.86370474   0.08445423  43.41318929  29.5400525
  28.80839951  35.49146213   9.94654135  45.44642111  49.92724101
  39.61010098   5.51640814  10.73777255   1.63731149  21.00114245] y: [ 10.82932219  21.02942356  11.40675322  44.36410545   1.98994692
  32.64703274   1.67373999  25.22054903  10.47671911  14.29969648
  23.96564805  17.4908811   19.63975846   7.80974962  42.9631454
   8.4657571   16.41816653  24.72689081  17.99215226  19.712853
  39.06103307  13.19203623  46.80627862  11.3538146   45.66007047
  29.22053168  33.8636191   16.59705599  31.10910258  42.99450883
  35.87999027  15.50410248  33.8602547   36.80669094  26.19652727
  41.03263082  48.52858159  46.7416254   12.79525051  40.93947387
  33.50004713   5.97955958  26.32420144  40.81463656  31.45289042
  11.90034695  32.17258227   1.31334136  44.77180661  42.89874867] z: [ 41.53156679  17.94271337  32.98051747  29.91283943  33.85258829
  29.28839641  35.07885361  42.96876258   7.54525312  39.73714955
   2.74072251  38.44423323  42.02539002  35.72564737  25.40085452
  19.45033564  14.19921361   3.65968781  10.04511327   8.22596127
  39.25294576  24.87387069  37.98585537  47.16644986  39.92862288
  11.75593985  36.83266165  19.04533176   0.17033388  29.80980027
   8.43266477  24.74634761  28.54004691  37.427965     8.11620562
  15.02601135  43.23526833  39.47980599  41.13746453  22.44735286
  19.35477562  13.34083578   7.82082924  37.98668125   3.44118681
  30.90037391  29.02449591   6.41743048  44.81353688]
54.0055466508

```
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/lagr2.png)  

# 7. Interpol_line.py  
Linear interpolation — interpolation by the algebraic binomial P₁ (x) = ax + b of the function f given at two points x₀ and x₁ of the segment [a, b]. If values ​​are specified at several points, the function is replaced by a piecewise linear function.  
Line 1 example  
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/line1.png)  
Line 2 example  
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/line2.png)  
# 8. Ur_tepla.py  
Other functions u 0 (x) and performance Methods μ. Give your feedback in the future Time T.  
∂u(x,t)/∂t - μ*(∂^2)(x,t)/∂x^2 = 0,x∈R,t>0   
u(x,0) = u0(x),x∈R. For example, u0(x) = 1/(2*x^2 + 1).      
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/ur_tepla.png)     
# 9. Ur_perenosa.py  
Extended function u 0 (x) and speed range C.  
Add a u (x, T) profile to the erased option Time T.  
∂u(x,t)/∂t + C*∂u(x,t)/∂x = 0, x ∈ R, t> 0    
u(x,0) = u0(x), x > 0  
u(0,t) = v0(t), t > 0      
![Image alt](https://github.com/Mcken09/Numerical-methods1/raw/master/pictures/ur_perenosa.png)   
# 10. Cholesky.py  
representation of a symmetric positive definite matrix A in the form A = L*(transpose(L)).
```
5
my: [[ 3.32876101  0.          0.          0.          0.        ]
 [ 3.19365873  2.3529401   0.          0.          0.        ]
 [ 2.34691053 -0.57243182  3.28690396  0.          0.        ]
 [ 4.19447816  0.77903347  0.92904603  4.22578097  0.        ]
 [ 4.76731311  2.83270142  1.25461321  3.19223037  1.01079774]]
linalg: [[ 3.32876101  0.          0.          0.          0.        ]
 [ 3.19365873  2.3529401   0.          0.          0.        ]
 [ 2.34691053 -0.57243182  3.28690396  0.          0.        ]
 [ 4.19447816  0.77903347  0.92904603  4.22578097  0.        ]
 [ 4.76731311  2.83270142  1.25461321  3.19223037  1.01079774]]
```
```
50
my: [[ 225.77280498    0.            0.         ...,    0.            0.            0.        ]
 [ 171.96251877  114.1671086     0.         ...,    0.            0.            0.        ]
 [ 153.5404154    61.74998421  103.01660587 ...,    0.            0.            0.        ]
 ..., 
 [ 141.86409759   50.46668138   25.33613292 ...,   28.39223084    0.            0.        ]
 [ 140.76541535   31.36852226   22.34845789 ...,  -14.11547875
     7.74053086    0.        ]
 [ 165.06597739   60.01865104   23.55898107 ...,   13.7694152   -27.73184074
    14.26941311]]
linalg: [[ 225.77280498    0.            0.         ...,    0.            0.            0.        ]
 [ 171.96251877  114.1671086     0.         ...,    0.            0.            0.        ]
 [ 153.5404154    61.74998421  103.01660587 ...,    0.            0.            0.        ]
 ..., 
 [ 141.86409759   50.46668138   25.33613292 ...,   28.39223084    0.            0.        ]
 [ 140.76541535   31.36852226   22.34845789 ...,  -14.11547875
     7.74053086    0.        ]
 [ 165.06597739   60.01865104   23.55898107 ...,   13.7694152   -27.73184074
    14.26941311]]


```










