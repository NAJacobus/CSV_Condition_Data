import math
def truncdec(real_num, sig):
    ##Truncates real numbers to sig non-zero decimal places
    if real_num == 0:
        return 0
    elif real_num >= 1:
        return round(real_num, sig)
    pv = math.floor(math.log10(abs(real_num)))
    return str(round(real_num, -1 * pv + sig - 1))



