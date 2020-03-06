
Is=0.01*10^-12;
Ib=0.1*10^-12;
Vb=1.3;
Gp=0.1;
vVec=linspace(-1.95,0.7,200);
iVec=zeros(1,200);

for x = 1:200
    %iVec(x) = Is*(exp(1.2.*vVec(x)/0.025)-1) + Gp*vVec(x) - Ib*(exp(-1.2*(vVec(x)+Vb)/0.025)-1);
    iVec(x) = Is.*(exp(1.2*vVec(x)/0.025)-1) + Gp.*vVec(x) - Ib*(exp(-1.2*((vVec(x)+Vb))/0.025)-1);
end

iVec2 = iVec + iVec.*(0.2*rand(1,200));

p4 = polyfit(vVec,iVec,4);
f4=polyval(p4,vVec);

p4i = polyfit(vVec,iVec,4);
f4i=polyval(p4i,vVec);

p8 = polyfit(vVec,iVec,8);
f8=polyval(p8,vVec);

p8i = polyfit(vVec,iVec,8);
f8i=polyval(p8i,vVec);

p4r = polyfit(vVec,iVec2,4);
f4r=polyval(p4r,vVec);

p4ri = polyfit(vVec,iVec2,4);
f4ri=polyval(p4ri,vVec);

p8r = polyfit(vVec,iVec2,8);
f8r=polyval(p8r,vVec);

p8ri = polyfit(vVec,iVec2,8);
f8ri=polyval(p8ri,vVec);

figure(1)
subplot(2,2,1)
plot(vVec,iVec)
title('No variation')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,f4)
plot(vVec,f8)
legend('Normal curve','4th order polynomial','8th order polynomial')
grid on
hold off

subplot(2,2,2)
semilogy(vVec,abs(iVec))
title('No variation')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,abs(f4i))
plot(vVec,abs(f8i))
legend('Normal curve','4th order polynomial','8th order polynomial')
grid on
hold off

subplot(2,2,3)
plot(vVec,iVec2)
title('20% random variation')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,f4r)
plot(vVec,f8r)
legend('Normal curve','4th order polynomial','8th order polynomial')
grid on
hold off

subplot(2,2,4)
semilogy(vVec,abs(iVec2))
title('20% random variation')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,abs(f4ri))
plot(vVec,abs(f8ri))
legend('Normal curve','4th order polynomial','8th order polynomial')
grid on
hold off

fo = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(vVec',iVec',fo)
If = ff(vVec')

foi = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffi = fit(vVec',iVec',foi)
Ifi = ff(vVec')

foii = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffii = fit(vVec',iVec',foii)
Ifii = ff(vVec')

forr = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffr = fit(vVec',iVec',forr)
Ifr = ff(vVec')

foir = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffir = fit(vVec',iVec',foir)
Ifir = ff(vVec')

foiir = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffiir = fit(vVec',iVec',foiir)
Ifiir = ff(vVec')

figure(2)
subplot(3,3,1)
plot(vVec,iVec,'*')
title('No variation, B and D set')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,If)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,2)
plot(vVec,iVec,'*')
title('No variation, D set')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,Ifi)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,3)
plot(vVec,iVec,'*')
title('No variation, Solving for all parameters (A,B,C,D)')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,Ifii)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,4)
semilogy(vVec,abs(iVec),'*')
title('No variation, B and D set')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,If)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,5)
semilogy(vVec,abs(iVec),'*')
title('No variation, D set')
xlabel('Voltage')
ylabel('Current')
hold on
semilogy(vVec,abs(Ifi))
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,6)
semilogy(vVec,abs(iVec),'*')
title('No variation, Solving for all parameters (A,B,C,D)')
xlabel('Voltage')
ylabel('Current')
hold on
semilogy(vVec,abs(Ifii))
legend('Normal curve','fit')
grid on
hold off

figure(3)
subplot(3,3,1)
plot(vVec,iVec2,'*')
title('20% variation, B and D set')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,Ifr)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,2)
plot(vVec,iVec2,'*')
title('20% variation, D set')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,Ifir)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,3)
plot(vVec,iVec2,'*')
title('20% variation, Solving for all parameters (A,B,C,D)')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,Ifiir)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,4)
semilogy(vVec,abs(iVec2),'*')
title('No variation, B and D set')
xlabel('Voltage')
ylabel('Current')
hold on
plot(vVec,Ifr)
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,5)
semilogy(vVec,abs(iVec2),'*')
title('No variation, D set')
xlabel('Voltage')
ylabel('Current')
hold on
semilogy(vVec,abs(Ifir))
legend('Normal curve','fit')
grid on
hold off

subplot(3,3,6)
semilogy(vVec,abs(iVec2),'*')
title('No variation, Solving for all parameters (A,B,C,D)')
xlabel('Voltage')
ylabel('Current')
hold on
semilogy(vVec,abs(Ifiir))
legend('Normal curve','fit')
grid on
hold off

% Neural Net Model 
inputs = vVec.';
targets = iVec.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs

figure(4)
subplot(2,2,1)
plot(vVec,iVec,'*')
hold on
plot(vVec,Inn)
hold off
grid on
subplot(2,2,2)
semilogy(vVec,abs(iVec),'*')
hold on
semilogy(vVec,abs(Inn))
hold off
grid on

inputs = vVec.';
targets = iVec2.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs

subplot(2,2,3)
plot(vVec,iVec2,'*')
hold on
plot(vVec,Inn)
hold off
grid on
subplot(2,2,4)
semilogy(vVec,abs(iVec2),'*')
hold on
semilogy(vVec,abs(Inn))
hold off
grid on