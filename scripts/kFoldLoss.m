function performance = kFoldLoss(x, y, cv, numHid, lr, mc)

% Train network.
net = patternnet(numHid, 'trainrp'); 
net.trainParam.lr = lr;
net.trainParam.mc = mc;
net.performParam.regularization = 0.1;
net = train(net, x(:,cv.training), y(:,cv.training));
% Evaluate on validation set and compute cross entropy
ypred = net(x(:, cv.test));
performance = perform(net,y(cv.test),ypred);
end




