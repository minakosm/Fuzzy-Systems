function [dh, dv] = getDistances(x,y)

x_flag = [(x>5) (x>6) (x>7)];
y_flag = [(y>1) (y>2) (y>3)];

for i=1:3
    if(x_flag(i) == 1 && y_flag(i) == 0)
        dh = NaN;
        dv = NaN;
        return;
    end
end

sumX = sum(x_flag);
sumY = sum(y_flag);

switch sumX
    case 3
        dh = 1;
        dv = y-3;
    case 2
        dv = y-2;
        if(sumY == 3)
            dh = 1;
        else
            dh = 7-x;
        end
    case 1
        dv = y-1;
        if(sumY == 3)
            dh = 1;
        elseif(sumY == 2)
            dh = 7-x;
        else 
            dh = 6-x;
        end
    case 0 
        dv = y;
        if (sumY == 3)
            dh = 1;
        elseif(sumY == 2)
            dh = 7-x;
        elseif(sumY == 1)
            dh = 6-x;
        else
            dh = 5-x;
        end
end

dh = min(1,dh);
dv = min(1,dv);


