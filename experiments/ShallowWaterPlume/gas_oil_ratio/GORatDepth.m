%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simplified calculation to convert the surface GOR to the GOR at a
% specified depth. 
% MATLAB code created September 04, 2012 by 
% Debra Simecek-Beatty, Physical Scientist
% National Oceanic and Atmosperic Administration
% 7600 Sand Point Way NE
% Seattle, WA 98115

% The total released volume of oil is strongly connected to the GOR or 
% gas-oil-ratio.  The GOR is the ratio of the volume of gas that comes out 
% of solution in standard conditions.  As the oil is brought to the 
% surface, natural gas comes out of the solution.  The user provides 
% the GOR which is typically valid at the surface.  The GOR is then
% adjusted for the sea floor (or the depth of the release).  The GOR is 
% typically reported as standard cubic feet per standard oil barrel. 
%
% The Pipeline Oil Spill Volume Estimator (OCS Study MMS 2002-033) has a 
% section for calculating the "GOR reduction factor".  The method uses a 
% series of tables in the calculations.  This technique was choosen to 
% calculate the "GOR reduction factor" as it appears to be a realiable 
% reference.  

% References: 
% OCS Study MMS 2002-033, "Pipeline Oil Spill Volume Estimator,
% Pocket Guide"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model assumptions
% Unfortunately, the units used in the pocket guide are not MKS.  This 
% means incoming MKS units need to be converted in order to use the tables
% in the guide book.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE       UNITS        DESCRIPTION
% GOR            scf/stb     Gas-oil-ratio in standard cubic feet per 
%                            standard oil barrel
% Psource        psi         Pipeline pressure pounds per square inch 
% d              feet        Water depth at rupture location feet (ft)
% t                          minutes (min)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last modified 04 September 2012 DSB
% clear variables and such NOTE - remove code when a function
clear all
clc
close all
commandwindow;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Start - Test Data in MKS Units
  
    d = 30;        % 30 m 
                   % equivalent to about 100 feet

    GOR = 2530;    % 2530 S m^3/ S m^3
                   % equivalant to 450 Scf/Stb
                   % (standard cubic feet/ stock tank barrel)

    % Start Calculate Pressure from release velocity
    Uo = 117.1;       % Velocity of the jet in meter per sec 
                      % 117.1 m/s is about 950 PSI
                      % 12.5 m/s is about 10 PSI

    Rhoi = 995.81;    % Jet density in kilogram per cubic meter MKS

    Psource = (1/2) * Rhoi * (Uo * Uo); % N/m^2  or Pa units (Pascals)
                                        % equavelent to 950 psi

    % End Calculate Pressure from release velocity    
%   End - Test Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%   Start - Convert MKS variables to units for the pocket guide
d = d * 3.2808;              % convert meter to feet

GOR = GOR * 5.6146;          % convert S m^3 / S m^3 to Scf/stb

Psource = Psource * (1/6897);  % convert Pa to psi
                               % JLM:
                               % should this be more accurate (6894.75)?
%   END - Convert MKS variables to units for the pocket guide
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Start-Equation 1.5, page 8
% Calculating ambient pressure outside leak at depth in psi
    Pamb = 0.446533*d;
% End-Equation 1.5

 % Start-Equation 1.4, page 8
 % The relative pressure, deltaPrel, difference over the leak point is 
     deltaPrel = Psource/Pamb;
 % End-Equation 1.4
 
% Start- Table 1.3, page 11
%: Maximum released volume fraction, frel
if (deltaPrel > 200)
    frel = 0.77;
    Gmax = 112;
elseif (deltaPrel < 1.1)
    frel = 0;
    Gmax = 0;
    error('deltaPrel is less than 1.1 therefore no leakage from Source');
else 
    % Relative Pressure ratio, deltaPrel
    TableOnePtThree.RelPRatioLow = [0 1.1 1.2 1.5 2 3 4 5 10 20 30 50];
    TableOnePtThree.RelPRatioHigh= [1 1.2 1.5 2 3 4 5 10 20 30 50 200];
    % Maximum release fraction, Maxfrel
    TableOnePtThree.Maxfrel = [0.0 0.08 0.17 0.30 0.40 0.47 0.50 0.55 0.64 0.71 0.74 0.76 0.77];
    % Maximum release occurs for a GOR of Gmax - Note that Gmax never > 560
    TableOnePtThree.maxG = [0 140 225 337 449 505 560 505 337 168 140 112 112];
 
    % Look in Table 1.3 and pick the frel and Gmax corresponding to 
    % relative pressure ratio.
    y =find(deltaPrel > TableOnePtThree.RelPRatioLow); %Find index of values that meet criteria
    frel = TableOnePtThree.Maxfrel(y(length(y)))       %The last index is the one needed
    Gmax = TableOnePtThree.maxG(y(length(y)))          % and will point to right spot on table
    clear y % temperary variable
end
% End-Table 1.3

% Start-Section 1.3.5
    % Table 1.4 GOR reduction factors, page 11
    % GOR < Gmax
    if (GOR < Gmax) 
        fGOR = GOR/Gmax % Column 2 of Table 1.4
    elseif GOR > Gmax
        % Gas-oil-ratio (GOR provided by user) reduction factor
        TableOnePtFour.GORLow = [0 225 280 340 420 560 1100 1700 2800 5600];
        TableOnePtFour.GORHigh = [225 280 340 420 560 1100 1700 2800 5600 113000];
        % GOR > Gmax % Column 3 of Table 1.4
        TableOnePtFour.GORgtGmax = [1 0.98 0.97 0.95 0.9 0.85 0.82 0.63 0.43 0.26];
        
        % Look in Table 1.4 and pick the Fgor value corresponding to GOR
        y =find(GOR > TableOnePtFour.GORLow); %Find index of values that meet criteria
        fGOR = TableOnePtFour.GORgtGmax(y(length(y))) %The last index is the one needed
                                                    % and will point to right spot on table
        clear y % temporary variable
    end
% End-Section 1.3.5

fGOR
% Start Section 1.3, Equation 1.1 page 5
% The total volume of oil, Vrel, is found from 
% Vrel = (0.1781 * Vpipe * frel * fGOR) + Vpreshut
% End Section 1.3, Equation Equation 1.1 page 5

