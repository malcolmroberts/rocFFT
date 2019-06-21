// Compute the ceiling of a / b.
int ceilquot(int a, int b)
{
  return (a + b - 1) # b;
}

// Compute the median of a sample.
real getmedian(real[] vals)
{
   vals = sort(vals);
   int half = quotient(vals.length, 2);
   return (vals.length % 2 == 0) 
      ? 0.5 * ( vals[half - 1] + vals[half] ) : vals[half];
   return 0;
}

real getmean(real[] vals)
{
   return sum(vals) / vals.length;
}

// Bootstrap resampling method for computing the 95% band for the
// median.
real[] mediandev(real[] vals)
{
    real[] medlh = new real[3];
   int nsample = vals.length;
   real resample[] = new real[nsample];
   for(int i = 0; i < nsample; ++i) {
      resample[i] = vals[i];
   }
   real median = getmedian(resample);
   medlh[0] = median;
   
   // Number of resamples to perform:
   int nperm = 200;
   real medians[] = new real[nperm];
   for(int i = 0; i < nperm; ++i) {
      for(int j = 0; j < nsample; ++j) {
	 resample[j] = vals[rand() % nsample];
      }
      medians[i] = getmedian(resample);
   }
   medians = sort(medians);
   real low = medians[(int)floor(nperm * 0.025)];
   real high = medians[(int)ceil(nperm * 0.975)];

   medlh[1] = low;
   medlh[2] = high;

   return medlh;
}
