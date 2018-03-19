function handle = plotCovariance(mean, covariance, confidence, varargin)
    % covariance = V * D * V'
    [V, D] = eig(covariance);
    
    sigma = sqrt(diag(D));
    
    scaling = sqrt(chi2inv(confidence, 2));
    
    if covariance(1, 1) > covariance(2, 2)
        phi = atan2(V(2, 2), V(1, 2));
        
        extent = scaling * [sigma(2) sigma(1)];
    else
        phi = atan2(V(2, 1), V(1, 1));
        
        extent = scaling * [sigma(1) sigma(2)];
    end
    
    handle = plotEllipse(mean, extent, phi, varargin{:});
end