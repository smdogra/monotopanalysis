#!/usr/bin/env python

variables = [ (x,'F') for x in 
      [
#        'tau32SD',
#        'qtau32',
#        'alphapull1',
#        'alphapull2',
#        'alphapull3',
#        'fitprob',
#        'avg_secfN_1_3_20',
      ]
]
#variables.append(['tau32SD','F','fj1Tau32SD'])
#variables.append(['htt_frec','F','fj1HTTFRec'])

formulae = [ (x,'F',x.replace('ecfN','fj1ECFN')) for x in 
    [   
        'ecfN_2_4_20/pow(ecfN_1_3_20,2.00)',
        'ecfN_2_4_10/pow(ecfN_1_3_10,2.00)',
        'ecfN_3_3_10/pow(ecfN_1_2_20,1.50)',
        'ecfN_2_4_10/ecfN_1_2_20',
        'ecfN_1_4_20/pow(ecfN_1_3_10,2.00)',
        'ecfN_2_3_05/pow(ecfN_1_2_20,0.50)',
        'ecfN_2_4_05/pow(ecfN_1_3_05,2.00)',
        'ecfN_1_4_10/pow(ecfN_1_3_05,2.00)',
        'ecfN_1_3_10/pow(ecfN_2_3_10,0.50)',
        'ecfN_1_3_10/ecfN_2_3_05',
        'ecfN_3_3_10/pow(ecfN_3_3_20,0.50)',
        'ecfN_1_3_10/pow(ecfN_3_3_10,0.33)',
        'ecfN_1_2_05/pow(ecfN_1_2_10,0.50)',
        'ecfN_1_3_20/ecfN_2_3_10',
        'ecfN_2_3_05/ecfN_1_3_10',
        'ecfN_3_3_20/pow(ecfN_3_3_10,2.00)',
        'ecfN_2_3_10/pow(ecfN_1_3_10,2.00)',
        'ecfN_2_3_10/ecfN_1_2_20',
        'ecfN_3_3_05/pow(ecfN_1_3_20,0.75)',
        'ecfN_1_2_10/pow(ecfN_1_2_05,2.00)',
        'ecfN_3_3_05/pow(ecfN_1_2_05,3.00)',
        'ecfN_2_3_10/ecfN_1_3_20',
        'ecfN_1_3_20/pow(ecfN_3_3_05,1.33)',
        'ecfN_1_2_10/pow(ecfN_1_2_20,0.50)',
        'ecfN_2_3_10/pow(ecfN_2_3_20,0.50)',
        'ecfN_3_3_20/pow(ecfN_2_3_20,1.50)',
        'ecfN_2_3_20/pow(ecfN_3_3_20,0.67)',
        'ecfN_1_2_20/pow(ecfN_1_2_10,2.00)',
        'ecfN_3_3_05/pow(ecfN_1_2_20,0.75)',
        'ecfN_1_3_20/ecfN_1_2_20',
        'ecfN_2_3_10/pow(ecfN_3_3_20,0.33)',
        'ecfN_2_3_10/pow(ecfN_3_3_10,0.67)',
        'ecfN_1_3_20/pow(ecfN_2_3_20,0.50)',
        'ecfN_1_3_20/pow(ecfN_3_3_20,0.33)',
        'ecfN_1_3_20/pow(ecfN_3_3_10,0.67)',
        'ecfN_1_3_10/pow(ecfN_2_3_20,0.25)',
        'ecfN_3_3_10/pow(ecfN_1_3_10,3.00)',
        'ecfN_3_3_10/pow(ecfN_1_3_20,1.50)',
        'ecfN_1_2_20/pow(ecfN_1_2_05,4.00)',
        'ecfN_2_3_20/pow(ecfN_1_3_20,2.00)',
        'ecfN_3_3_10/pow(ecfN_2_3_10,1.50)',
        'ecfN_1_3_10/pow(ecfN_3_3_05,0.67)',
        'ecfN_3_3_20/pow(ecfN_1_3_20,3.00)',
        'ecfN_1_2_05/pow(ecfN_1_2_20,0.25)',
        'ecfN_3_3_05/pow(ecfN_1_3_10,1.50)',
        'ecfN_1_3_10/pow(ecfN_1_2_20,0.50)',
        'ecfN_1_3_10/pow(ecfN_3_3_20,0.17)',
        'ecfN_3_3_20/pow(ecfN_2_3_10,3.00)',

    ]
]

spectators = [
#          ('tau32','F'),
          ('pt','F'),
          ('mSD','F'),
              ]
