"""

 2020 (c) piteren

    some basic / useful decorators

"""

from inspect import getfullargspec, ismethod, isfunction
import time

# decorator printing time report
def timing(f):
    def new_f(*args, **kwargs):
        stime = time.time()
        f(*args, **kwargs)
        print(f'(@timing) {f.__name__} finished, taken {time.time() - stime:.1f}sec')
    new_f.__name__ = f'{f.__name__}:@timing'
    return new_f

# prints debug info about parametersand given args/kwargs of function/method
def args(f):
    def new_f(*args, **kwargs):
        ins = getfullargspec(f)
        no_val = '--'

        arL = ins.args
        if arL[0] == 'self': arL = arL[1:]              # simple, BUT not 100% accurate, ...but who names param 'self'?

        defL = list(ins.defaults)                       # list of default values
        defL = [no_val]*(len(arL)-len(defL)) + defL     # pad them with '--' from the beginning
        arL = [list(e) for e in zip(arL,defL)]          # zip together

        # append args from the beginning
        if args:
            arvL = args[1:]
            for ix in range(len(arvL)):
                v = arvL[ix]
                arL[ix].append(v)

        kwL = [[k,kwargs[k]] for k in kwargs]           # kwargs in list

        # get from kwargs params of f, append their values and remove from kwargs
        kremIXL = []
        for ix in range(len(kwL)):
            for e in arL:
                if kwL[ix][0]==e[0]:
                    kremIXL.append(ix)
                    e.append(kwL[ix][1])
        for ix in reversed(kremIXL): kwL.pop(ix)

        # add no value to not overridden params
        for e in arL:
            if len(e)<3: e.append(no_val)

        # calc columns widths
        kw = [10,10,10]
        for e in arL:
            for i in range(3):
                lse = len(str(e[i]))
                if lse>kw[i]: kw[i] = lse
        for e in kwL:
            for i in range(2):
                lse = len(str(e[i]))
                ki = i if not i else 2
                if lse > kw[ki]: kw[ki] = lse
        kw = [w+2 for w in kw]

        print(f'\n@args report of:  {f.__name__:40s} ********************************************')
        print(f' > {"param":{str(kw[0])}s} {"default":{str(kw[1])}s} {"given":{str(kw[2])}s}')
        for e in arL: print(f'   {str(e[0]):{str(kw[0])}s} {str(e[1]):{str(kw[1])}s} {str(e[2]):{str(kw[2])}s}')
        print(f' > **kwargs (not used by {f.__name__}):')
        for e in kwL: print(f'   {str(e[0]):{str(kw[0])}s} {"":{str(kw[1])}s} {str(e[1]):{str(kw[2])}s}')
        print('@args report finished *********************************************************************************\n')

        f(*args, **kwargs)

    new_f.__name__ = f'{f.__name__}:@args'
    return new_f


def example_args():

    class TestArgs:

        @args
        def do_something(
                self,
                parameter_long_name_a,
                pa_b,
                pa_c,
                def_a=  1,
                def_b=  31.2434,
                **kwargs):

            print(parameter_long_name_a,pa_b,pa_c,def_a,def_b)

    exob = TestArgs()
    exob.do_something(10, 11, pa_c=12, def_b=5, oth_a=6, first=0.000025)


if __name__ == "__main__":

    example_args()