import graph;
import utils;
import stats;

//asy datagraphs -u "xlabel=\"\$\\bm{u}\cdot\\bm{B}/uB$\"" -u "doyticks=false" -u "ylabel=\"\"" -u "legendlist=\"a,b\""

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

//scale(Linear,Linear);
scale(Log, Log);
//scale(Log,Linear);

bool dolegend = true;

string filenames = getstring("filenames");
string legendlist = "";

real xmin = -inf;
real xmax = inf;

bool doxticks = true;
bool doyticks = true;
string xlabel = "Problem size";
string ylabel = "Time, [s]";

bool normalize = false;

bool raw = false;

usersetting();

bool myleg = ((legendlist == "") ? false: true);
string[] legends=set_legends(legendlist);

if(normalize) {
   scale(Log, Linear);
   ylabel = "Time / problem size, [s]";
}

bool plotxval(real x) {
   return x >= xmin && x <= xmax;
}

int n = -1;
bool flag = true;
int lastpos;
while(flag) {
   ++n;
   int pos=find(filenames,",",lastpos);
   string filename;
   if(lastpos == -1) {filename = ""; flag=false;}
   filename = substr(filenames,lastpos,pos-lastpos);

   if(flag) {
      write(filename);
      lastpos = pos > 0 ? pos + 1 : -1;

      real[] x;
      real[] y;
      real[] ly;
      real[] hy;

      if(!raw) {
	 file fin=input(filename).line();
	 real[][] a = fin.dimension(0,0);
	 a = transpose(a);
	 x = a[0];
	 y = normalize ? a[1] / x : a[1];
	 ly = normalize ? a[2] / x : a[2];
	 hy = normalize ? a[3] / x : a[3];
      } else {
	 bool moretoread = true;
	 file fin = input(filename);
	 while(moretoread) {
	    int a = fin;
	    if(a == 0) {
	       moretoread = false;
	       break;
	    } 
	    x.push(a);

	    int n = fin;
	    real vals[] = new real[n];
	    for(int i = 0; i < n; ++i) {
	       real val = fin;
	       if(normalize)
		  vals[i] = val / a;
	       else
		  vals[i] = val;
	    }
	    if(a >= xmin && a <= xmax) {
	       real median, low, high;
	       mediandev(vals, median, low, high);
	       y.push(median);
	       ly.push(low);
	       hy.push(high);
	    }
	 }
      }
     
      pen p = Pen(n);
      if(n == 2)
	 p = darkgreen;

      pair[] z;
      pair[] dp;
      pair[] dm;
      for(int i = 0; i < x.length; ++i) {
	 if(plotxval(x[i])) {
	    z.push((x[i] , y[i]));
	    dp.push((0 , y[i] - hy[i]));
	    dm.push((0 , y[i] - ly[i]));
	 }
      }
      errorbars(z, dp, dm, p);

      if(n == 1) 
	 p += dashed;
      if(n == 2) 
	 p += Dotted;
    
      guide g = scale(0.5mm) * unitcircle;
      marker mark = marker(g, Draw(p + solid));

      bool drawme[] = new bool[x.length];
      for(int i = 0; i < drawme.length; ++i) {
	 drawme[i] = true;
	 if(!plotxval(x[i]))
	    drawme[i] = false;
	 if(y[i] <= 0.0)
	    drawme[i] = false;
      }

      draw(graph(x, y, drawme), p,  
	   myleg ? legends[n] : texify(filename), mark);
   }
}

if(doxticks)
   xaxis(xlabel,BottomTop,LeftTicks);
else
   xaxis(xlabel);

if(doyticks)
   yaxis(ylabel,LeftRight,RightTicks);
else
   yaxis(ylabel,LeftRight);

if(dolegend)
   attach(legend(),point(plain.E),20plain.E);
