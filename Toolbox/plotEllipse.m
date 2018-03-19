function handle = plotEllipse(center, extent, angle, varargin)
    a = 0:0.01:2*pi;
    
    s = [extent(1) * cos(a)
         extent(2) * sin(a)];
    
    ca = cos(angle);
    sa = sin(angle);
    
    s = [ca -sa
         sa  ca] * s;
    
    handle = plot([s(1, :) s(1, 1)] + center(1), [s(2, :) s(2, 1)] + center(2), varargin{:});
end