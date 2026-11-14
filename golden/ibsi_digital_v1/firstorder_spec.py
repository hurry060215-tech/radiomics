"""Independent IBSI first-order arithmetic from exact original-phantom histogram.

This is a Golden specification only, never a runtime backend. Fractions define
population central moments and linear quantiles without native/PyRadiomics calls.
"""
from fractions import Fraction
import math


def calculate(histogram):
    data=sorted((int(gray),int(count)) for gray,count in histogram.items())
    if not data or any(count<=0 for _,count in data):
        raise ValueError("Histogram must contain positive integer counts.")
    n=sum(count for _,count in data)
    mean=Fraction(sum(gray*count for gray,count in data),n)
    moments={power:sum((count*(Fraction(gray)-mean)**power for gray,count in data),Fraction(0))/n for power in [2,3,4]}
    values=[gray for gray,count in data for _ in range(count)]
    def quantile(p):
        position=(n-1)*p
        lo=position.numerator//position.denominator
        hi=(position.numerator+position.denominator-1)//position.denominator
        return Fraction(values[lo])+(position-lo)*(values[hi]-values[lo])
    energy=sum(gray*gray*count for gray,count in data)
    variance=moments[2]
    excess=moments[4]/variance**2-3 if variance else Fraction(0)
    mad=sum((count*abs(Fraction(gray)-mean) for gray,count in data),Fraction(0))/n
    uniformity=sum((Fraction(count,n)**2 for _,count in data),Fraction(0))
    rational={"Mean":mean,"Median":quantile(Fraction(1,2)),"Minimum":Fraction(values[0]),"Maximum":Fraction(values[-1]),
              "10Percentile":quantile(Fraction(1,10)),"90Percentile":quantile(Fraction(9,10)),"Range":Fraction(values[-1]-values[0]),
              "InterquartileRange":quantile(Fraction(3,4))-quantile(Fraction(1,4)),"Variance":variance,"MeanAbsoluteDeviation":mad,
              "Kurtosis":excess,"Energy":Fraction(energy),"Uniformity":uniformity}
    output={name:float(value) for name,value in rational.items()}
    output.update(Skewness=float(moments[3])/float(variance)**1.5 if variance else 0.,
                  RootMeanSquared=math.sqrt(energy/n),Entropy=-sum(float(Fraction(count,n))*math.log2(float(Fraction(count,n))) for _,count in data))
    return {"roi_count":n,"intensity_sum":sum(gray*count for gray,count in data),"sum_of_squares":energy,
            "histogram":[{"gray":gray,"count":count} for gray,count in data],
            "rational_reference":{name:{"numerator":value.numerator,"denominator":value.denominator} for name,value in rational.items()},
            "central_moment_fractions":{str(power):{"numerator":value.numerator,"denominator":value.denominator} for power,value in moments.items()},
            "ibsi_features":output,"native_raw_kurtosis_definition":float(moments[4]/variance**2) if variance else 0.,
            "exact_integer_features":[name for name,value in rational.items() if value.denominator==1]}
