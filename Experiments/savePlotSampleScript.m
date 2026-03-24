set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultLineLineWidth', 2)
set(groot, 'defaultLineMarkerSize', 8)
set(groot, 'defaultAxesFontSize', 22)
set(groot, 'defaultTextFontSize', 22)
set(groot, 'defaultFigureColor', 'w')
set(groot, 'defaultFigurePosition', [200 200 600 400])
set(groot, 'defaultLegendInterpreter', 'latex')
set(groot, 'defaultTextInterpreter', 'latex')

savePlots = 1;

x = 0:0.01:1;
y1 = x.^2+ sin(x.^2);
y2 = x.^2+0.05;

figure
plot(x,y1,'-')
hold on
plot(x,y2,'--')
legend({'$y_1$','$y_2$'},'Location','northwest','Fontsize',25)
legend boxoff
xlabel('This is $x$')
ylabel('This is $y$')

 if savePlots
    print(gcf,'-depsc','testPlot') % save eps plot
    %print(gcf,'-depsc2','testPlot2') % save eps plot
    print(gcf,'-dpng','testPlot') % save png plot
 end