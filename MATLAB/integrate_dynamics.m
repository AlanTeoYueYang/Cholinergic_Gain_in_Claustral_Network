%Tried to make everything TanH
function [output] = integrate_dynamics(W, params, x0)

    [output.t, X] = ode45(@rate_dynamics_ode, ...
        linspace(0, params.tfinal, params.n_timepoints), x0, [], W, params);

    % Convert neuronal activities into firing rates
    output.r = mixed_gain_2(X, params.ACh, params.r0);
    
    output.X = X;

end

function x_dot = rate_dynamics_ode(t, X, W, params)
I = normrnd(10,sqrt(5),size(X));
if params.with_stim == 1
    if (t >= 500) && (t <= 600)  
        I = I + 200;
    elseif (t >= 1500) && (t <= 1600)
        I = I + 400;
    end
elseif params.with_stim == 2
    I = normrnd(200,200,size(X));
end
    x_dot = params.over_tau*(-X+I+W*mixed_gain(X,params.ACh,params.r0));
end

% Gain function
function out_X = mixed_gain(X, ACh,r0)
    CC = X(1:180);
    CS = X(181:270);
    VIP = X(271:280);
    SST = X(281:290);
    PV = X(291:300);
    
    if ACh == 0
        CC_out = gain_fn(CC,r0,44.1,0.3243);
        CS_out = gain_fn(CS,r0,45.38,0.2092);
        VIP_out = gain_fn(VIP,r0,48.21,0.2799);
    else
        CC_out = gain_fn(CC,r0,43.53,0.2098);
        CS_out = gain_fn(CS,r0,40.68,0.3221);
        VIP_out = gain_fn(VIP,r0,48.92,1.117);
    end
    
    SST_out = gain_fn(SST,r0, 71.53,0.536);
    PV_out = gain_fnPV(PV,r0,100,1.826);
    out_X = [CC_out;CS_out;VIP_out;SST_out;PV_out];
end

function out = gain_fn(x, r0, rmax, g)
    out = zeros(size(x));
    for n = 1 : length(x)
        if x(n) < 0
        out(n) = r0*tanh(g*x(n)/r0); 
        else
        out(n) = (rmax-r0)*tanh(g*x(n)/(rmax-r0));
        end
    end
end

function out = gain_fnPV(x, r0, rmax, g)
    out = zeros(size(x));
    x1 = x - 140;
    for n = 1 : length(x1)
        if x1(n) < 0
        out(n) = r0*tanh(g*x1(n)/r0); 
        else
        out(n) = (rmax-r0)*tanh(g*x1(n)/(rmax-r0));    
        end      
    end
end

function out_X = mixed_gain_2(X, ACh, r0)
    CC = X(:,1:180);
    CS = X(:,181:270);
    VIP = X(:,271:280);
    SST = X(:,281:290);
    PV = X(:,291:300);
    
    if ACh == 0
        CC_out = gain_fn_2(CC,r0,44.1,0.3243);
        CS_out = gain_fn_2(CS,r0,45.38,0.2092);
        VIP_out = gain_fn_2(VIP,r0,48.21,0.2799);
    else
        CC_out = gain_fn_2(CC,r0,43.53,0.2098);
        CS_out = gain_fn_2(CS,r0,40.68,0.3221);
        VIP_out = gain_fn_2(VIP,r0,48.92,1.117);
    end
    
    SST_out = gain_fn_2(SST,r0, 71.53,0.536);
    PV_out = gain_fn_2PV(PV,r0,100,1.826);
    out_X = [CC_out,CS_out,VIP_out,SST_out,PV_out];
end

function out = gain_fn_2(x, r0, rmax, g)
    out = zeros(size(x));
    for row = 1 : size(x,1)
        for col = 1 : size(x,2)
            if x(row,col) < 0
              out(row,col) =  r0*tanh(g*x(row,col)/r0);
            else
              out(row,col) = (rmax-r0)*tanh(g*x(row,col)/(rmax-r0));    
            end
                       
        end
    end
end

function out = gain_fn_2PV(x, r0, rmax, g)
    out = zeros(size(x));
    x1 = x - 140;
    for row = 1 : size(x1,1)
        for col = 1 : size(x1,2)
            if x1(row,col) < 0
              out(row,col) =  r0*tanh(g*x1(row,col)/r0);
            else
              out(row,col) = (rmax-r0)*tanh(g*x1(row,col)/(rmax-r0));    
            end     
        end
    end
end