function example(varargin)

%# define defaults at the beginning of the code so that you do not need to
%# scroll way down in case you want to change something or if the help is
%# incomplete
options = struct('noiseType',{'No','Log-normal','Bernoulli','Markovian'},...
       'delayTypes',{'no','no','no','no'},...
       'updateP',[1,1,1,1]);

%# read the acceptable names
optionNames = fieldnames(options);

%# count arguments
if length(varargin) > 1
  nArgs = length(varargin(2:end));
  if round(nArgs/2)~=nArgs/2
    error('EXAMPLE needs propertyName/propertyValue pairs')
  end
end

for pair = reshape(varargin(2:end),2,[]) %# pair is {propName;propValue}
   inpName = pair{1}; %# make case insensitive

   if any(strcmp(inpName,optionNames))
      %# overwrite options. If you want you can test for the right class here
      %# Also, if you find out that there is an option you keep getting wrong,
      %# you can use "if strcmp(inpName,'problemOption'),testMore,end"-statements
      options(1).(inpName) = pair(2);
   else
      error('%s is not a recognized parameter name',inpName)
   end
end
end