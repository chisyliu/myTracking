clear
clc
close all

% Instantiate system model
sysModel = TargetSysModel();

% Instantiate measurement Model
measModel = PolarMeasModel();

% The estimator
filter = SIRPF();
filter.setNumParticles(10^5);

% Initial state estimate
initialState = Gaussian([1 1 0 0 0]', [10, 10, 1e-1, 1, 1e-1]);
filter.setState(initialState);

% Simulate system state trajectory and noisy measurements
numTimeSteps = 100;

sysStates = nan(5, numTimeSteps);
measurements = nan(2, numTimeSteps);
updatedStateMeans = nan(5, numTimeSteps);
updatedStateCovs = nan(5, 5, numTimeSteps);
predStateMeans = nan(5, numTimeSteps);
predStateCovs = nan(5, 5, numTimeSteps);
runtimesUpdate = nan(1, numTimeSteps);
runtimesPrediction = nan(1, numTimeSteps);

sysState = initialState.drawRndSamples(1);

for k = 1:numTimeSteps
    % Simulate measurement for time step k
    measurement = measModel.simulate(sysState);
    
    % Save data
    sysStates(:, k)    = sysState;
    measurements(:, k) = measurement;
    
    % Perform measurement update
    runtimesUpdate(:, k) = filter.update(measModel, measurement);
    
    [updatedStateMeans(:, k), ...
        updatedStateCovs(:, :, k)] = filter.getStateMeanAndCov();
    
    %% TEST
    % resampling
    dm = filter.getState();
    dm_copy = dm.copy();
    [samples, weights] = dm_copy.getComponents();
    indx = resampleSystematic(weights);
    rndSamples = samples(:, indx);
    % Gaussian mixture learning
    numOfCompos = 7;
    GMModel = fitgmdist(rndSamples', numOfCompos, 'RegularizationValue', 0.1);
    % Gaussian mixture fusion
    % TODO
    
    %%
    
    % Simulate next system state
    sysState = sysModel.simulate(sysState);
    
    % Perform state prediction
    runtimesPrediction(:, k) = filter.predict(sysModel);
    
    [predStateMeans(:, k), ...
        predStateCovs(:, :, k)] = filter.getStateMeanAndCov();
    
end

close all;

% Plot state prediction and measurement update runtimes
figure();
hold on;
grid  on;
xlabel('Time step');
ylabel('Runtime in ms');
color  = filter.getColor();
name   = filter.getName();

r = runtimesUpdate(1, :) * 1000;
plot(1:numTimeSteps, r, 'DisplayName', sprintf('Update %s', name) );

r = runtimesPrediction(1, :) * 1000;
plot(1:numTimeSteps, r, 'DisplayName', sprintf('Prediction %s', name) );

set(gca, 'yscale', 'log');

legend show;

% Plot state estimates
figure();

hold on;
axis equal;
grid on;
xlabel('x');
ylabel('y');

name   = filter.getName();

title(['Estimate of ' name]);

% Plot true system state
plot(sysStates(1, :), sysStates(2, :), 'k-', 'LineWidth', 2);

% Show confidence interval of 99%
confidence = 0.99;

objectTrace = nan(2, 2 * numTimeSteps);
j = 1;

for k = 1:numTimeSteps
    % Plot updated estimate
    updatedPosMean = updatedStateMeans(1:2, k);
    updatedPosCov  = updatedStateCovs(1:2, 1:2, k);
    
    plotCovariance(updatedPosMean, updatedPosCov, confidence, 'b-');
    
    objectTrace(:, j) = updatedPosMean;
    j = j + 1;
    
    % Plot predicted estimate
    predPosMean = predStateMeans(1:2, k);
    predPosCov  = predStateCovs(1:2, 1:2, k);
    
    plotCovariance(predPosMean, predPosCov, confidence, 'r-');
    
    objectTrace(:, j) = predPosMean;
    j = j + 1;
end

% Plot object trace
plot(objectTrace(1, :), objectTrace(2, :), 'Color', [0 0.5 0], 'LineWidth', 1);
