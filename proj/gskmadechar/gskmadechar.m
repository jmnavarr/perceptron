function varargout = gskmadechar(varargin)
% GSKMADECHAR M-file for gskmadechar.fig
%      GSKMADECHAR, by itself, creates a new GSKMADECHAR or raises the existing
%      singleton*.
%
%      H = GSKMADECHAR returns the handle to a new GSKMADECHAR or the handle to
%      the existing singleton*.
%
%      GSKMADECHAR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GSKMADECHAR.M with the given input arguments.
%
%      GSKMADECHAR('Property','Value',...) creates a new GSKMADECHAR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before gskmadechar_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to gskmadechar_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help gskmadechar

% Last Modified by GUIDE v2.5 23-Mar-2008 23:58:49

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gskmadechar_OpeningFcn, ...
                   'gui_OutputFcn',  @gskmadechar_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before gskmadechar is made visible.
function gskmadechar_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gskmadechar (see VARARGIN)

% Choose default command line output for gskmadechar
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gskmadechar wait for user response (see UIRESUME)
% uiwait(handles.figure1);
clc
clear global
global inmv000 wtmv000 incmv000 iolmv000
inmv000 = zeros(100,1);
inmv000 = (-1) + inmv000;
incmv000 = zeros(1,100);
incmv000 = (-1) + incmv000;
iolmv000 = 45;
wtmv000 = zeros(100,100);
wtmv000 = wtmv000 + (inmv000 * inmv000');
%for intv011 = 1:100
%    wtmv000(intv011,intv011) = 0;
%end







% --- Outputs from this function are returned to the command line.
function varargout = gskmadechar_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global wtmv000 inmv000 incmv000 iolmv000
inmv001 = zeros(100,1);
inmv001 = (-1) + inmv001;
chrv000 = cd;
chrv001 = get(handles.edit1,'string');
chrv002 = get(handles.edit3,'string');
chrvv002 = str2double(chrv002);
dos('%SystemRoot%\system32\mspaint.exe gskchar.bmp');
un8v000 = imread(fullfile(chrv000,'gskchar.bmp'));
%calculation of x1, y1, x2, y2
[intv000,intv001,intv002] = size(un8v000);
intv005 = 0;
intv006 = 0;
intv007 = 0;
intv008 = 0;
intv009 = intv000;
intv010 = intv001;
for intv003 = 1:intv000
    intv007 = 0;
    for intv004 = 1:intv001
        if (un8v000(intv003,intv004,1) ~= 255) || (un8v000(intv003,intv004,2) ~= 255) || (un8v000(intv003,intv004,3) ~= 255)
            intv007 = intv007 + 1;
            if intv007 == 1
                intv009 = min(intv003,intv009);
                intv010 = min(intv004,intv010);
            end
            intv005 = max(intv005,intv003);
            intv006 = max(intv006,intv004);
        end
    end
end
if ((intv005 == intv009) || (intv006 == intv010)) || (intv005 == 0) || (intv006 == 0) || (intv009 == 0) || (intv010 == 0)
    msgbox([{'Image cannot have less than 2 lines (line must be longer than 2 pixels)',char(13),'Retrieval is not sucessful'}],'ChrRecg - GSK');
else
    if (intv005 - intv009) > (intv006 - intv010)
        intv003 = 9 / (intv005 - intv009);
    else
        intv003 = 9 / (intv006 - intv010);
    end
    axes(handles.axes1);
%    figure
    cla
    hold on
    for intv011 = 1 : intv000
        for intv012 = 1 : intv001
            if (un8v000(intv011,intv012,1) ~= 255) || (un8v000(intv011,intv012,2) ~= 255) || (un8v000(intv011,intv012,3) ~= 255)
                intv013 = round(intv003 * (intv011 -intv009)) + 1;
                intv014 = round(intv003 * (intv012 -intv010)) + 1;
                inmv001((intv013-1) * 10 + intv014,1) = 1;
                plot(intv014,-intv013);
            end
        end
    end
    xlim([0 11]);
    ylim([-11 0]);
    hold off
    logv000 = 0;
    intv004 = 0;
    intv005 = 0;
    [intv000,intv001] = size(incmv000);
    opmv000 = inmv001;
    while logv000 == 0
        intv004 = intv004 + 1;
        intv016(1:intv000,1) = zeros(intv000,1);
        for intv002 = 1 : intv000
            intv015(1:100,1) = incmv000(intv002,1:100) - opmv000(1:100,1)';
            intv016(intv002,1) = norm(intv015(1:100,1));
        end
        intv017 = min(intv016);
        intv018 = 0;
        for intv002 = 1 : intv000
            if intv016(intv002,1) == intv017;
                intv018 = intv018 + 1;
                intv003 = intv002;
            end
        end
        if (intv018 == 1)
            logv000 = 1;
        end
        inmmv001 = opmv000;
        if logv000 ~= 1
            if intv004 > 100
                intv004 = intv004 - 100;
            end
            intv005 = intv005 + 1;
            if get(handles.checkbox2,'value') == 1
                inmv001(intv004,1) = opmv000(intv004,1);
            else
                inmv001 = opmv000;
            end
        end
        if intv005 == chrvv002 * 10000
            msgbox([{'Processing taking too long',char(13),'Retrieval is not sucessful'}],'ChrRecg - GSK');
            logv000 = 1;
            intv003 = 1;
        end
        opmv000 = wtmv000' * inmv001;
        for intv002 = 1 : 100
            if opmv000(intv002) < 0
                opmv000(intv002) = -1;
            else
                opmv000(intv002) = 1;
            end
        end
    end
    set(handles.edit4,'string',num2str(intv005));
    chrv001 = strcat(chrv001,char(iolmv000(intv003)));
    set(handles.edit1,'string',chrv001);
end
intv000 = get(handles.checkbox1,'value');
intvs000 = num2str(intv000);
if intvs000 == '1'
    gskmadechar('pushbutton1_Callback',hObject, eventdata, handles)
end



% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.edit1,'string','');

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global inmv000 wtmv000 incmv000 iolmv000
[chrv002,chrv003]=uiputfile('*.mat','Select a File to store Data');
if ~(num2str(chrv002) == '0');
    save(fullfile(chrv003,chrv002));
    msgbox('Saving is sucessful','ChrRecg - GSK');    
else
    msgbox([{'You Have Specify File To Save Data',char(13),'Saving is not sucessful'}],'ChrRecg - GSK');
end


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[chrv005,chrv006]=uigetfile('*.mat','Select a File to Load Data');
if (num2str(chrv005) == '0');
    msgbox([{'You Have Specify File To Save Data',char(13),'Loading is not sucessful'}],'ChrRecg - GSK');
else
    chrv007 = questdlg([{'This will delete present learned data!!!'};{'Are You Sure ???'}],'Question ?','Yes','No.','No.','Yes');
    if chrv007 == 'Yes'
        chrv008 = handles;
        load(fullfile(chrv006,chrv005));
    else
        msgbox([{'Cancelled by user',char(13),'Loading is not sucessful'}],'ChrRecg - GSK');
    end
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global wtmv000 inmv000 incmv000 iolmv000
inmv000 = zeros(100,1);
inmv000 = (-1) + inmv000;
chrv000 = cd;
chrv001 = get(handles.edit2,'string');
dos('%SystemRoot%\system32\mspaint.exe gskchar.bmp');
un8v000 = imread(fullfile(chrv000,'gskchar.bmp'));
%calculation of x1, y1, x2, y2
[intv000,intv001,intv002] = size(un8v000);
intv005 = 0;
intv006 = 0;
intv007 = 0;
intv008 = 0;
intv009 = intv000;
intv010 = intv001;
for intv003 = 1:intv000
    intv007 = 0;
    for intv004 = 1:intv001
        if (un8v000(intv003,intv004,1) ~= 255) || (un8v000(intv003,intv004,2) ~= 255) || (un8v000(intv003,intv004,3) ~= 255)
            intv007 = intv007 + 1;
            if intv007 == 1
                intv009 = min(intv003,intv009);
                intv010 = min(intv004,intv010);
            end
            intv005 = max(intv005,intv003);
            intv006 = max(intv006,intv004);
        end
    end
end
if ((intv005 == intv009) || (intv006 == intv010)) || (intv005 == 0) || (intv006 == 0) || (intv009 == 0) || (intv010 == 0)
    msgbox([{'Image cannot have less than 2 lines (line must be longer than 2 pixels)',char(13),'Teaching is not sucessful'}],'ChrRecg - GSK');
else
    if (intv005 - intv009) > (intv006 - intv010)
        intv003 = 9 / (intv005 - intv009);
    else
        intv003 = 9 / (intv006 - intv010);
    end
    axes(handles.axes1);
%    figure
    cla
    hold on
    for intv011 = 1 : intv000
        for intv012 = 1 : intv001
            if (un8v000(intv011,intv012,1) ~= 255) || (un8v000(intv011,intv012,2) ~= 255) || (un8v000(intv011,intv012,3) ~= 255)
                intv013 = round(intv003 * (intv011 -intv009)) + 1;
                intv014 = round(intv003 * (intv012 -intv010)) + 1;
                inmv000((intv013-1) * 10 + intv014,1) = 1;
                plot(intv014,-intv013);
            end
        end
    end
    xlim([0 11]);
    ylim([-11 0]);
    hold off
    [intv000,intv001] = size(incmv000);
    if intv000>0
        logv000 = 0;
        for intv002 = 1 : intv000
            if inmv000(1:100,1) == incmv000(intv002,1:100)'
                logv000 = 1;
                if ~isempty(chrv001)
                    intv004 = unicode2native(chrv001);
                    iolmv000(intv002,1)= intv004(1,1);
                    logv001 = 1;
                end
            end
        end
    end
    if (intv000 == 0) | (logv000 == 0)
        incmv000(intv000+1,1:100) = inmv000;
        intv004 = unicode2native(chrv001);
        iolmv000(intv000+1,1) = intv004(1,1);
        wtmv000 = wtmv000 + (inmv000 * inmv000');
%        for intv011 = 1:100
%            wtmv000(intv011,intv011) = 0;
%        end
    end
end


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
chrv000 = [{'CHARACTER RECOGNITION WITH NEURAL ANALYSIS'}...
    {''}...
    {}...
    {'This programme used simple algorithm of auto assosiative principle of artificial neural networks analysis with asynchronous updation. Algorithm used here is very simple and robust one.'}...
    {''}...
    {''}...
    {'Number of iterations required depend on the mach between input given to the available teaching data. Number of iteration may exceen with improper teaching. We allow you to specify time limit of each processing so that incase of long iteration the processing will stop to reduce the problem of hanging.'}...
    {''}...
    {''}...
    {'First we need to train the system by typing alphabet or numaric in textbox provided.'}...
    {''}...
    {'Paintbrush will open where we can draw input image, use "Air Brush" tool in paintbrush for good results'}...
    {''}...
    {'Same alphabet can be taught more than once (It is recomended to teach each alphabet more than 10 times'}...
    {''}...
    {'You can save the systems learned data for future use or for future updation. It is similar to stoping work now and continuing afterwards. Load / Save buttons should be used for this purpose'}...
    {''}...
    {'By pressing the button start, you are allowed to enter the image which will approximated to nearest alphabet of numaric'}...
    {''}...
    {'Please send your suggetions to "gadisureshkumar@gmail.com"'}...
    {''}...
    {''}...
    {'Prepared by Suresh Kumar Gadi'}];
msgbox(chrv000,'ChrRecg - GSK');
    
    
    
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

chrv000 = questdlg([{'This will delete present learned data!!!'};{'Are You Sure ???'}],'Question ?','Yes','No.','No.','Yes');
if chrv000 == 'Yes'
    clear global
    global inmv000 wtmv000 incmv000 iolmv000

    inmv000 = zeros(100,1);
    inmv000 = (-1) + inmv000;
    incmv000 = zeros(1,100);
    incmv000 = (-1) + incmv000;
    iolmv000 = 45;
    wtmv000 = zeros(100,100);
    wtmv000 = wtmv000 + (inmv000 * inmv000');
%    for intv011 = 1:100
%        wtmv000(intv011,intv011) = 0;
%    end
else
    msgbox([{'Cancelled by user',char(13),'Learned data clearing is not sucessful'}],'ChrRecg - GSK');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global inmv000 wtmv000 incmv000 iolmv000
intv000 = wtmv000 / max(abs(max(abs(wtmv000))));
figure
hold on
grid off
for intv001 = 1:100
    for intv002 = 1:100
        if intv000(intv001,intv002) < 0
            plot(intv001,100-intv002,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[-1*intv000(intv001,intv002) 0 0],'MarkerSize',10);
        else
            if intv000(intv001,intv002) > 0
                plot(intv001,100-intv002,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[0 intv000(intv001,intv002) 0],'MarkerSize',10);
            else
                plot(intv001,100-intv002,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[1 1 0],'MarkerSize',10);
            end
        end
    end
end
hold off
        
% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global inmv000 wtmv000 incmv000 iolmv000
[intv000, intv001] = size(incmv000);
figure
hold on
for intv002 = 1:intv000
    for intv003 = 1:100
        if incmv000(intv002,intv003) > 0
            plot(intv003,intv002,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[0 1 0],'MarkerSize',10);
        else
            plot(intv003,intv002,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[1 0 0],'MarkerSize',10);
        end
    end
end
hold off        
%msgbox([{'Recorded input patterens are displyed on mainwindow of MATLAB'}],'ChrRecg - GSK');


% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global inmv000 wtmv000 incmv000 iolmv000
char(iolmv000)
msgbox([{'Recorded Ouputs are displyed on mainwindow of MATLAB'}],'ChrRecg - GSK');





function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1




% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global inmv000 wtmv000 incmv000 iolmv000
figure
hold on
for intv000 = 1:100
    if inmv000(intv000,1) > 0
        plot(intv000,1,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[0 1 0],'MarkerSize',10);
    else
        plot(intv000,1,'--rs','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor',[1 0 0],'MarkerSize',10);
    end
end
hold off
%msgbox([{'Present input matrix is displyed on mainwindow of MATLAB'}],'ChrRecg - GSK');


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2




% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


