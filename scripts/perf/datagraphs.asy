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

string filenames = "";
string legendlist = "";

real xmin = -inf;
real xmax = inf;

bool doxticks = true;
bool doyticks = true;
string xlabel = "Problem size";
string ylabel = "Time [s]";

bool normalize = false;

bool raw = true;

bool speedup = true;

usersetting();
write("filenames:\"", filenames+"\"");

if(filenames == "")
    filenames = getstring("filenames");

bool myleg = ((legendlist == "") ? false: true);
string[] legends=set_legends(legendlist);

if(normalize) {
   scale(Log, Linear);
   ylabel = "Time / problem size, [s]";
}

bool plotxval(real x) {
   return x >= xmin && x <= xmax;
}

string[] listfromcsv(string input)
{
    string list[] = new string[];
    int n = -1;
    bool flag = true;
    int lastpos;
    while(flag) {
        ++n;
        int pos = find(input, ",", lastpos);
        string found;
        if(lastpos == -1) {
            flag = false;
            found = "";
        }
        found = substr(input, lastpos, pos - lastpos);
        if(flag) {
            list.push(found);
            lastpos = pos > 0 ? pos + 1 : -1;
        }
    }
    return list;
}

string[] testlist = listfromcsv(filenames);

real[][] x = new real[testlist.length][];
real[][] y = new real[testlist.length][];
real[][] ly = new real[testlist.length][];
real[][] hy = new real[testlist.length][];
real[][][] data = new real[testlist.length][][];
real xmax = 0.0;
real xmin = inf;

for(int n = 0; n < testlist.length; ++n)
{
    string filename = testlist[n];

    data[n] = new real[][];
    write(filename);

    real[] ly;
    real[] hy;

    int dataidx = 0;
    
    bool moretoread = true;
    file fin = input(filename);
    while(moretoread) {
        int a = fin;
        if(a == 0) {
            moretoread = false;
            break;
        } 
        
        int N = fin;
        if (N > 0) {
            xmax = max(a,xmax);
            xmin = min(a,xmin);

            x[n].push(a);

            data[n][dataidx] = new real[N];
            
            real vals[] = new real[N];
            for(int i = 0; i < N; ++i) {
                vals[i] = fin;
                data[n][dataidx][i] = vals[i];
            }
            //if(a >= xmin && a <= xmax) {
            real[] medlh = mediandev(vals);
            y[n].push(medlh[0]);
            ly.push(medlh[1]);
            hy.push(medlh[2]);
            //}
            ++dataidx;
        }
    }
   
    pen p = Pen(n);
    if(n == 2)
        p = darkgreen;

    pair[] z;
    pair[] dp;
    pair[] dm;
    for(int i = 0; i < x[n].length; ++i) {
        if(plotxval(x[n][i])) {
            z.push((x[n][i] , y[n][i]));
            dp.push((0 , y[n][i] - hy[i]));
            dm.push((0 , y[n][i] - ly[i]));
        }
    }
    errorbars(z, dp, dm, p);

    if(n == 1) 
        p += dashed;
    if(n == 2) 
        p += Dotted;
    
    guide g = scale(0.5mm) * unitcircle;
    marker mark = marker(g, Draw(p + solid));

    bool drawme[] = new bool[x[n].length];
    for(int i = 0; i < drawme.length; ++i) {
        drawme[i] = true;
        if(!plotxval(x[n][i]))
	    drawme[i] = false;
        if(y[n][i] <= 0.0)
	    drawme[i] = false;
    }

     
    draw(graph(x[n], y[n], drawme), p,  
         myleg ? legends[n] : texify(filename), mark);
}

if(doxticks)
   xaxis(xlabel,BottomTop,LeftTicks);
else
   xaxis(xlabel);

if(doyticks)
    yaxis(ylabel,speedup ? Left : LeftRight,RightTicks);
else
   yaxis(ylabel,LeftRight);


if(dolegend)
    attach(legend(),point(plain.E),(speedup ? 60*plain.E + 40 *plain.N : 20*plain.E)  );


if(speedup) {
    string[] legends = listfromcsv(legendlist);
    // TODO: error bars
    // TODO: when there is data missing at one end, the axes might be weird

    picture secondary=secondaryY(new void(picture pic) {
            scale(pic,Log,Linear);
            real ymin = inf;
            real ymax = -inf;
            for(int n = 0; n < testlist.length; n += 2)
            {
                real[] xval = new real[];
                real[] yval = new real[];
                pair[] zy;
                pair[] dp;
                pair[] dm;
                for(int i = 0; i < x[n].length; ++i) {
                    for(int j = 0; j < x[n+1].length; ++j) {
                        if (x[n][i] == x[n+1][j]) {
                            xval.push(x[n][i]);
                            real val = y[n][i] / y[n+1][j];
                            yval.push(val);

                            zy.push((x[n][i], val));
                            real[] lowhi = ratiodev(data[n][i], data[n+1][j]);
                            real hi = lowhi[1];
                            real low = lowhi[0];

                            dp.push((0 , hi - val));
                            dm.push((0 , low - val));
    
                            ymin = min(val, ymin);
                            ymax = max(val, ymax);
                            break;
                        }
                    }
                    
                }
                if(xval.length > 0){
                    pen p = black+dashed;
                    if(n == 2) {
                        p = black + Dotted;
                    }
                    guide g = scale(0.5mm) * unitcircle;
                    marker mark = marker(g, Draw(p + solid));
                
                    draw(pic,graph(pic,xval, yval),p,legends[n] + "/" + legends[n+1],mark);
                    errorbars(pic, zy, dp, dm, p);
                }

                {
                    real[] fakex = {xmin, xmax};
                    real[] fakey = {ymin, ymax};
                    // draw an invisible graph to set up the limits correctly.
                    draw(pic,graph(pic,fakex, fakey),invisible);

                }
                yequals(pic, 1.0, lightgrey);
            }
            yaxis(pic,"speedup",Right,  black,LeftTicks);
            attach(legend(pic),point(plain.E), 60*plain.E - 40 *plain.N  );
        });
    

    add(secondary);
}

