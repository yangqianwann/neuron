%label generator with index
close all  %input 264*264 image
clear all
[a,aa]= uigetfile('*','multiselect','on');
output_size=256;
sample_size=1;
N=numel(a)/2;
Blabel=false(N*sample_size,1,output_size,output_size);
Wlabel=zeros(N*sample_size,1,output_size,output_size);
Centroid=nan(N*sample_size,200,2);
images=zeros(N*sample_size,1,output_size,output_size,'uint16');
positions=zeros(N*sample_size,3);
tic
for i=1:N
    i
    b=a(2*i-1:2*i);
    mat=contains(b,'.mat');
    tif=contains(b,'.tif');
    load(b{mat})
    I=imread(b{tif});
    if ~exist('roi_list','var')
        roi_list=ROI_list;
    end
    n=numel(roi_list);
    [s1,s2]=size(I);
    L=false(s1,s2);
    SL=zeros(s1,s2);
    La=zeros(n,2);
    for j=1:n
        l=L;
        l(roi_list(j).pixel_idx)=true;
        L=L|l;  %L is the binary label
        %
        [a1,a2]=find(roi_list(j).perimeter==1);
        La(j,1)=mean(a1);
        La(j,2)=mean(a2);
        %
    end
    for j=1:sample_size  %1000
        flag=5;
        while flag~=0
            c1=randi(s1-output_size+1);  %s1,s2=1152  output_size=256
            c2=randi(s2-output_size+1);
            r1=c1:c1+output_size-1;
            r2=c2:c2+output_size-1; %r1 and r2 are the range of each box
            test=false(s1,s2);
            test(r1(1),r2)=true;
            test(r1(end),r2)=true;
            test(r1,r2(1))=true;
            test(r1,r2(end))=true; 
            Test=test&(~L);
            if sum(sum(Test))>700 && sum(sum(L(r1,r2)))>3000 
                %first condition means the boarder of box can't cross many
                %ROIs, the second condition means the box has to include
                %enough ROIs
                flag=0;
            end
        end
        a1=I(r1,r2);
        b1=a1>5000;
        a1(b1)=5000;
        images(j+(i-1)*sample_size,1,:,:)=a1;
        Blabel(j+(i-1)*sample_size,1,:,:)=L(r1,r2);
        positions(j+(i-1)*sample_size,:)=[r1(1),r2(1),i];
        S=[];
        for k=1:n
            %{
            [a1,a2]=find(roi_list(k).perimeter==1);
            a1=mean(a1);
            a2=mean(a2);
            %}
            a1=La(k,1);
            a2=La(k,2);
            lb=[r1(1)-6.5,r2(1)-6.5];
            ub=[r1(end)+6.5,r2(end)+6.5];
            if sum(lb<[a1,a2])==2 && sum([a1,a2]<=ub)==2
                S=[S;k,a1,a2];
            end
        end
        [~,idx]=sort(S(:,3));
        idx=S(idx,1);
        centroid=[];
        for k=1:numel(idx)
            SL(roi_list(idx(k)).pixel_idx)=k;
            [a1,a2]=find(roi_list(idx(k)).perimeter==1);
            if mean(a1)-r1(1)+0.5>0 && mean(a2)-r2(1)+0.5>0
                centroid=[centroid;mean(a1)-r1(1)+0.5,mean(a2)-r2(1)+0.5];
            end
        end
        Wlabel(j+(i-1)*sample_size,1,:,:)=SL(r1,r2);
        Centroid(j+(i-1)*sample_size,1:numel(centroid)/2,:)=centroid;
    end
end

toc         
            
save('Test.mat','images','Blabel','Wlabel','Centroid')         
    