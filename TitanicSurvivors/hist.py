
import sys

try: 
    import pylab as plt
except:
    try: 
        from matplotlib import pylab as plt
    except:
        import sys
        sys.exit(-1)
        pass
    pass

plt.ioff()

def hist(df, value, condition='', bins=10):
    plt.clf()
    try : x=df[df.eval(condition)][value].values
    except : x=df[value].values
    plt.hist(x, bins=bins)
    plt.xlabel(value)
    plt.savefig(value+'_hist.png')

def efficiency(num, den, bins=10, xlabel='', ylabel='', fname='test.png'):
    h_den=plt.hist(den,bins=bins)
    h_num=plt.hist(num,bins=bins)

    plt.clf()

    for i in range(0, len(h_num[0])):
        if( h_den[0][i] == 0 ) :
            h_num[0][i] = 0
        else :            
            h_num[0][i] /= h_den[0][i]
            pass

        continue
    x_axis=[]
    for i in range(0, len(h_num[1])-1 ):
        x_axis.append( (h_num[1][i]+h_num[1][i+1])/2 )
        continue

    plt.plot(x_axis, h_num[0],'b.',ms=10)
    plt.xlim(min(x_axis)*0.8, max(x_axis)*1.2)
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print('file name is',fname)
    plt.savefig(fname)
    return h_num
