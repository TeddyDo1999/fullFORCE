import math 
import numpy as np
import scipy.stats as sp

def fullFORCE(mode,g,N,p,ran,TRLS,Tinit,task,hint,V,*argv):
# function varargout = fullFORCE(mode,g,N,p,ran,TRLS,Tinit,task,hint,V,varargin)
# Inputs:
# mode: for training, or testing
# g: various gain parameters   --- type: dict. g.keys() = [r, fout, fin, fhint]
# N: network size parameters
# p: an eclectic group of parameters, related to time, etc.
# TRLS: how to long to do RLS training for
# Tinit: how long to drive the system before learning
# task: name of the task
# hint: use a hint or not
# V: eigenvectors for the target-generating network, (possible) learned in a previous function

# Output: 
# mode == train: learned parameters from RLS
# mode == test: nMSError
    if len(argv) > 0:
        lrn = argv[1]
    
    #unpack parameters
    gr = g['r']
    gfout = g['fout']
    gfin = g['fin']
    gfhint = g['fhint']


    taux = p['taux']
    dt = p['dt']
    eta = p['eta']
    etaux = math.exp(-dt/taux)

    NN = N['N']
    Nout = N['out']

    J = ran['J']
    ufout = ran['fout']
    ufin = ran['fin']
    ufhint  = ran['fhint']

    DTRLS = p['DTRLS']

    #Line handles to be put here

    #Initialize parameters for RLS
    learnTargetDim = V.shape[1] # Nn in MATAB code

    if str(mode) == 'train':
        w = np.zeros([learnTargetDim, NN])
        W = np.zeros([Nout, NN])
        PS = 1*np.eye(NN)
    elif str(mode) == 'test':
        w = lrn['w']
        W = lrn['W']

    Jr = np.dot(gr, J) #gr is scalar, J is scalar * square matrix size N

    #apply gain/scaling to fout, fin and fhint (scalar mult)
    ufout = gfout*ufout
    ufin = gfin*ufin
    ufhint = gfhint*ufhint

    #target-generating network state
    xTeach = 1e0 * np.random.normal(size=[NN, 1])
    #task-performing network state
    xLearn = xTeach
    #state vector
    z = np.zeros(learnTargetDim)

    ttrial = np.inf #time index
    TTrial = 0 #trial length
    EZ = 0 #
    NF = 0
    ttime = 0 #timer using numTrials completed
    
    #Simulation
    while (ttime <= (TRLS + Tinit)) :
        if ttrial >= float(TTrial): #new trial starts
            ttrial = 1
            if task == 'ready_set_go':
                [fin,fout,fhint,TTrial,ITI] = trial_ready_set_go(p, hint)
            
            #compute the inputs to each unit
            Fout = np.dot(ufout,fout)
            Fin = np.dot(ufin, fin)
            Fhint = np.dot(ufhint, fhint)
            
            Zs = np.zeros([Nout,int(TTrial)]) #to accumulate generated outputs, for plotting
            
            ttime = ttime + 1 #increment trial counter by 1

        #target generating network    
        f = np.dot(Jr,np.tanh(xTeach)) + Fout[:, ttrial] + Fhint[:,ttrial]
        xinf = f + Fin[:,ttrial]
        xTeach = xinf + np.dot((xTeach - xinf),etaux)
        f = np.dot(np.transpose(V), f)

        #full-FORCE task-performing network
        temp = np.dot(V, z) + Fin[:, ttrial] - xLearn
        temp2 = np.sqrt(np.dot(eta, dt))
        xLearn = xLearn + dt/taux + temp + np.dot(temp2, np.random.normal(size=[NN, 1]))
        r = np.tanh(xLearn)

        #compute feedback from learning
        if ttime > Tinit:
            Z = np.dot(W, r)
            z = np.dot(w, r)
        else: #initialization hasn't elapsed, used target
            z = f
            Z = fout[0][ttrial] 

        #generated output for plotting
        Zs[0][ttrial] = Z

        #do RLS
        if (((np.random.uniform() < 1/DTRLS) & (ttime > Tinit)) & (mode == 'train')):
            xP = np.dot(PS, r)
            k = (1 + np.dot(np.transpose(r), xP))/np.transpose(xP)
            PS = PS - np.dot(xP, k)
            w = w - np.dot((z-f), k)
            W = W - np.dot((Z - fout[ttrial]), k)

        if ttrial == int(TTrial)-1:
            #plot here 
            if NF == 0:
                print("%s fullFORCE \nfullFORCE Error: %s\n%g trials of %g \n" %(mode, 'NaN', ttime, (TRLS+Tinit)))
            else:
                print("%s fullFORCE \nfullFORCE Error: %g\n%g trials of %g \n" %(mode, 100 * EZ/NF, ttime, (TRLS+Tinit)))

        if ttime > Tinit:
            EZ_change = (Z - fout[0][ttrial])
            EZ = EZ + np.dot(EZ_change, EZ_change.transpose())
            NF = NF + np.dot(fout[0][ttrial], fout[0][ttrial].transpose())
        
        ttrial = ttrial +1

    if mode == 'train':
        lrn['w'] = w
        lrn['W'] = W
        lrn['PS'] = PS
        return lrn
    elif mode == 'test':
        ERR = 100*EZ/NF
        return ERR        

    #delete line handles of plots here






def trial_ready_set_go(p, hint):
    ITIb = 0.4 #min ITI time 
    ITIl = 2.0 #mean ITI time
    ITI = 2 * round(0.5 * ((ITIb + np.random.exponential(ITIl)) * (1/p['dt'])))
    
    delay = round(np.random.uniform(low=10, high=210)*p['taux'], 3) 
    #pick random pulse interval

    event = np.round(np.divide([0.05, delay, 0.05, delay, 0.5],p['dt']))
    #timing of event sequence, for each trial
    '''imagine event = [A B C D E] '''
    
    TTrial = sum(event) + ITI #time of current trial

    IOITI = np.zeros([1,int(ITI/2)]) #empty vector
    I2ITI = -0.5 * np.ones([1, int(ITI/2)]) #inter-trial input

    IO1 = np.ones([1, int(event[0])]) #pulse inputs
    IO2_1 = delay * np.linspace(0, 1, event[1] + event[2]/2) #hint upwards
    IO2_2 = delay * np.linspace(1, 0, event[3] + event[2]/2) #hint downwards

    #turn off hint signal
    if hint == 'nohint':
        IO2_1 = 0 * IO2_1
        IO2_2 = 0 * IO2_2

    #output
    BetaCurve = sp.beta.pdf(np.linspace(0, 1, event[4]), 4, 4)
    maximaBetaCurve = max(BetaCurve)

    normalized_betaCurve = BetaCurve*(1/maximaBetaCurve)
    IO3 = -0.5 + 1.5 * normalized_betaCurve
    #target output

    #construct inputs, hints and outputs from above signals

    #task input 
    sum1 = int(sum(event[1::]))
    sum2 = int(sum(event[0:2]))
    sum3= int(sum(event[3::]))

    #fin = np.concatenate((IOITI, IO1, np.zeros(sum1), IOITI), axis=0) 
    #fin_temp = np.concatenate((IOITI, np.zeros(sum2), IO1, np.zeros(sum3), IOITI), axis=0)
    fin_1 = np.concatenate((IOITI, IO1, np.zeros([1,sum1]), IOITI), axis=1)
    fin_2 = np.concatenate((IOITI, np.zeros([1,sum2]), IO1, np.zeros([1,sum3]), IOITI), axis=1)
    

    fin = np.concatenate((fin_1, fin_2), axis=0) 
        
    ''' row 1: padding with 0, pulse (1) through event 1 (pulse), 0 through delay-pulse-delay-off, padding 0
    row 2:padding0, zeros through pulse-delay,  pulse, zeros delay-off, padding 0
    
    fin = [padding0 A=1 B=0 C=0 D=0 E=0 padding0;
           padding0 A=0 B=0 C=1 D=0 E=0 padding0]
    '''

    #desired output signals, and hint signals
    IO2_1 = IO2_1.reshape((1, len(IO2_1)))
    IO2_2 = IO2_2.reshape((1, len(IO2_2)))
    fhint = np.concatenate((IOITI, np.zeros([1, int(event[1])]), IO2_1, IO2_2, np.zeros([1, int(event[4])]), IOITI), axis=1)
    
    '''
    % fint is padding time (zeros for 1/2 no of dt between each trial) then
    % zeros for first event elements (the pulse input) then IO2_1 and IO2_2
    % replace event 2, 3, 4 which are the delay-pulse(=0.05)-delay (delay is
    % random by each trial, then zros for last event (Long pulse of 0.5) then
    % padding again

    %fhint = [padding0 A=0 B-C/2=0-delay C/2-D=delay-0 E=0 padding0]
    '''
    IO3 = IO3.reshape((1, len(IO3)))
    fout = np.concatenate((I2ITI, -0.5 * np.ones([1, int(sum(event[0:4]))]), IO3, I2ITI), axis=1)
    '''
    %Padding with -0.5 for inter-trial-input, then -0.5 for events 1 to 4
    %(pulse delay pulse delay), then output vector, then padding with -0.5

    %fout = [padding-0.5 A=-.5 B=-.5 C=-.5 D=-.5 E=IO3 padding-0.5]

    %all the padding has duration = ITI/2
    '''
    return [fin,fout,fhint,TTrial,ITI] #fin, fout, fhint are lists instead of ndarray


def trial_ready_set_go_NOHINT(p):

    ITIb = 0.4
    ITIl = 2.0
    
    delay = round(np.random.uniform(low=10, high=210)*p['taux'], 3)


    event = np.dot([0.05, 14.95], (1/p['dt']))
    TTrial=sum(event)
    
    IO1= np.ones(1,event[0])

    # simulate with expected output cosine: 
    #IO3 = cos(2*pi*5.*linspace(0,TTrial/2-1, TTrial/2)*p.dt);
    IO3 = (.7)*math.sin(math.pi*np.linspace(start=0, stop=(TTrial/2 -1), num=TTrial/2)*p['dt'])
    IO3 = np.concatenate([IO3, -1*np.fliplr(IO3)], axis=1) #can use np.flip(IO3, 1), it's the same
    #IO3 is 0.7sin(pi*t) with t from (0 to TTrial)
    IO1_filler = np.zeros(1,event[1]) #### CHECK THIS LINE
    #Original code have zeros(1, sum(event(2))), 2 in matlab is 1 in python.
    # event.shape = (1,2). so event[1] is scalar. sum(scalar) in matlab returns
    # that scalar, while sum(scalar) in python returns error.
    # Why was sum(event(2)) used in the orig matlab code?    
    fin = np.concatenate([IO1, IO1_filler], axis=1)
    fout = IO3
    pass
     



if __name__ == "__main__":
    pass
    #fullFORCE(mode,g,N,p,ran,TRLS,Tinit,task,hint,V,'something')    