%% petrov_galerkin_final_REALLY_FIXED.m
% Key fixes:
%  (A) Force plot box to fill axes (no "pushed down" / shrinking)
%  (B) Put equation box close (not far right)
%  (C) Right-angle marker + a short local residual arrow at same base point
% -------------------------------------------------------------------------
clear; close all; clc;

%% ===== Style ============================================================
LW_curve3D = 3.2;
LW_curve2D = 3.0;
LW_dash    = 1.6;
LW_axis    = 1.8;
LW_vec     = 3.8;
LW_box     = 2.2;

axisBlue   = [0.08 0.20 0.88];
trialColor = [0.10 0.28 0.92];
testColor  = [0.10 0.60 0.20];
resRed     = [0.85 0.06 0.06];
planeCol   = [0.91 0.91 0.91];

FS = 19;

%% ===== Smooth 3D trajectory =============================================
t  = linspace(0, 2*pi, 500);

a1 = 0.50 + 0.55*cos(t)     + 0.14*cos(2.1*t);
a2 = 0.40 + 0.80*sin(0.9*t) + 0.10*sin(3.0*t);
a3 = 1.20 + 0.38*sin(1.4*t) + 0.10*cos(2.5*t);

a1p = a1;  a2p = a2;  a3p = zeros(size(a1));

%% ===== Evaluation point =================================================
[~, k0] = max(a3(50:350));
k0 = k0 + 49;
x_full = [a1(k0); a2(k0); a3(k0)];

% f(x)
VSCALE  = 1.5;
fx_dir  = [0.88; -0.18; 0.50];
fx_dir  = fx_dir / norm(fx_dir);
fx      = VSCALE * fx_dir;

%% ===== OBLIQUE PROJECTION ===============================================
Ur = [1 0; 0 1; 0 0];                 % trial space: z=0 plane

w1 = [1.0;  0.25;  0.55];  w1 = w1/norm(w1);
w2 = [0.10; 1.00;  0.45];  w2 = w2/norm(w2);
Wr = [w1, w2];

G = Wr' * Ur;
if rcond(G) < 1e-10
    error('Wr^T Ur ill-conditioned. Choose different w1,w2.');
end

P_obliq = Ur * (G \ Wr');             % oblique projector

fx_proj  = P_obliq * fx;
r        = fx - fx_proj;

tip_f  = x_full + fx;
tip_Pf = x_full + fx_proj;

%% ===== Limits ===========================================================
planePad = 0.18;
xL_plane = [min(a1)-planePad,  max(a1)+planePad];
yL_plane = [min(a2)-planePad,  max(a2)+planePad];

xpad = 0.65;  ypad = 0.65;  zpad = 0.75;
xL   = [min(a1)-xpad,  max([a1(:)', tip_f(1), tip_Pf(1)])+xpad];
yL   = [min(a2)-ypad,  max([a2(:)', tip_f(2), tip_Pf(2)])+ypad];

a3_axlen = max(a3(:)) + 0.25;
zL       = [-0.10,  max([a3(:)', tip_f(3)])+zpad];

%% ===== Figure ============================================================
figure('Color','w','Units','pixels','Position',[80 80 1500 780]);
hold on;

ax = gca;
ax.Units = 'normalized';
ax.Position = [0.04 0.06 0.74 0.90];       % BIG main plot
ax.ActivePositionProperty = 'position';

% --- REAL FIX for "pushed down / shrunk": force plot box fill -------------
ax.DataAspectRatioMode      = 'manual';
ax.PlotBoxAspectRatioMode   = 'manual';
ax.PlotBoxAspectRatio       = [1.55 1.0 0.85];  % tweak if needed
% -------------------------------------------------------------------------

%% ===== Trial plane =======================================================
[xp,yp] = meshgrid(xL_plane, yL_plane);
surf(xp, yp, zeros(2,2), ...
     'FaceColor',planeCol,'FaceAlpha',0.55, ...
     'EdgeColor',[0.50 0.50 0.50],'LineWidth',1.0);

text(mean(xL_plane)-0.3, yL_plane(2)-0.25, 0.05, ...
     '$\mathcal{U}_r$ (trial space)', ...
     'FontSize',FS-2,'Interpreter','latex', ...
     'Color',[0.30 0.30 0.30],'FontAngle','italic');

%% ===== Coordinate axes ===================================================
O      = [xL_plane(1),  yL_plane(1),  0       ];
ax2end = [xL_plane(2)+0.10, yL_plane(1), 0    ];
ax1end = [xL_plane(1),  yL_plane(2)+0.10, 0   ];
ax3end = [xL_plane(1),  yL_plane(1),  a3_axlen];

plot3([O(1) ax2end(1)],[O(2) ax2end(2)],[0 0],            'k','LineWidth',LW_axis);
plot3([O(1) ax1end(1)],[O(2) ax1end(2)],[0 0],            'k','LineWidth',LW_axis);
plot3([O(1) ax3end(1)],[O(2) ax3end(2)],[O(3) a3_axlen],  'k','LineWidth',LW_axis);

text(ax2end(1)+0.07, ax2end(2),      0,             '$a_2$','FontSize',FS+1,'Interpreter','latex','Color',axisBlue);
text(ax1end(1),      ax1end(2)+0.07, 0,             '$a_1$','FontSize',FS+1,'Interpreter','latex','Color',axisBlue);
text(ax3end(1)-0.20, ax3end(2),      a3_axlen+0.04, '$a_3$','FontSize',FS+1,'Interpreter','latex','Color',axisBlue);

%% ===== 2D projected trajectory ==========================================
plot3(a1p, a2p, a3p, 'Color',trialColor,'LineWidth',LW_curve2D);

%% ===== Vertical drop lines ==============================================
dropIdx = round(linspace(40, 460, 10));
for k = dropIdx
    plot3([a1(k) a1(k)],[a2(k) a2(k)],[0 a3(k)],'--', ...
          'Color',[0.38 0.38 0.38],'LineWidth',LW_dash);
end

%% ===== 3D trajectory =====================================================
plot3(a1, a2, a3, 'k','LineWidth',LW_curve3D);
plot3(a1(1),a2(1),a3(1),'ko','MarkerFaceColor','k','MarkerSize',8);

%% ===== Evaluation point ==================================================
plot3(x_full(1),x_full(2),x_full(3),'ko','MarkerFaceColor','k','MarkerSize',10);
text(x_full(1)-0.40, x_full(2)-0.05, x_full(3)+0.09, ...
     '$x(t)$','FontSize',FS,'Interpreter','latex','Color','k');

%% ===== Test-space vectors (W_r) =========================================
wLen = 0.90;

quiver3(x_full(1),x_full(2),x_full(3), wLen*w1(1),wLen*w1(2),wLen*w1(3), 0, ...
        'Color',testColor,'LineWidth',2.4,'MaxHeadSize',0.7,'LineStyle','--');
quiver3(x_full(1),x_full(2),x_full(3), wLen*w2(1),wLen*w2(2),wLen*w2(3), 0, ...
        'Color',testColor,'LineWidth',2.4,'MaxHeadSize',0.7,'LineStyle','--');

%% ===== Vectors f, Pf, r =================================================
quiver3(x_full(1),x_full(2),x_full(3), fx(1),fx(2),fx(3), 0, ...
        'Color','k','LineWidth',LW_vec,'MaxHeadSize',0.55);
text(tip_f(1)+0.06, tip_f(2)+0.02, tip_f(3)+0.07, ...
     '$f(x)$','FontSize',FS+1,'Interpreter','latex','Color','k','FontWeight','bold');

quiver3(x_full(1),x_full(2),x_full(3), fx_proj(1),fx_proj(2),fx_proj(3), 0, ...
        'Color',trialColor,'LineWidth',LW_vec,'MaxHeadSize',0.55);
text(tip_Pf(1)+0.06, tip_Pf(2)-0.07, tip_Pf(3)-0.20, ...
     '$P_r f(x)$','FontSize',FS+1,'Interpreter','latex','Color',trialColor,'FontWeight','bold');

% residual drawn as difference tip_Pf -> tip_f
quiver3(tip_Pf(1),tip_Pf(2),tip_Pf(3), (tip_f(1)-tip_Pf(1)),(tip_f(2)-tip_Pf(2)),(tip_f(3)-tip_Pf(3)), 0, ...
        'Color',resRed,'LineWidth',LW_vec,'MaxHeadSize',0.55);
text(tip_Pf(1)+0.45*(tip_f(1)-tip_Pf(1)), ...
     tip_Pf(2)+0.45*(tip_f(2)-tip_Pf(2)), ...
     tip_Pf(3)+0.45*(tip_f(3)-tip_Pf(3)), ...
     '$r$','FontSize',FS+4,'Interpreter','latex','Color',resRed,'FontWeight','bold');

plot3([tip_f(1) tip_Pf(1)],[tip_f(2) tip_Pf(2)],[tip_f(3) tip_Pf(3)], ...
      '--','Color',[0.45 0.55 0.95],'LineWidth',LW_dash+0.4);

text(tip_Pf(1)-0.55, tip_Pf(2)+0.18, tip_Pf(3)+0.10, ...
     '(oblique)','FontSize',FS-3,'Color',[0.45 0.45 0.45],'FontAngle','italic');

%% ===== TRUE visual proof: r ⟂ w1 at SAME point ==========================
w1_unit = w1/norm(w1);
r_unit  = r/norm(r);

% choose a base point on the w1 ray from x_full
ra_base = x_full + 0.55*wLen*w1_unit;

% draw a SHORT local r-direction arrow from the SAME base point
r_loc_len = 0.35;  % purely visual
quiver3(ra_base(1),ra_base(2),ra_base(3), r_loc_len*r_unit(1),r_loc_len*r_unit(2),r_loc_len*r_unit(3), 0, ...
        'Color',resRed,'LineWidth',2.6,'MaxHeadSize',0.7,'LineStyle','--');

% right-angle marker using w1 and local r
sq = 0.14;
ra_pts = [ra_base, ...
          ra_base + sq*w1_unit, ...
          ra_base + sq*(w1_unit + r_unit), ...
          ra_base + sq*r_unit, ...
          ra_base];
plot3(ra_pts(1,:),ra_pts(2,:),ra_pts(3,:), '-', 'Color',testColor,'LineWidth',LW_box);

text(ra_base(1)+0.10, ra_base(2)+0.05, ra_base(3)+0.10, ...
     '$r \perp \mathcal{W}_r$','FontSize',FS-1,'Interpreter','latex', ...
     'Color',resRed,'FontWeight','bold');

%% ===== View =============================================================
xlim(xL); ylim(yL); zlim(zL);
axis off; grid off;
axis vis3d;
view(132, 26);
camproj('perspective');
camzoom(1.35);        % make it BIG

%% ===== Equation box (close, not far right) ==============================
annotation('textbox',[0.79 0.20 0.20 0.62], ...
    'String',{ ...
      '\bf Petrov--Galerkin:', ...
      ' ', ...
      '$\mathcal{W}_r \neq \mathcal{U}_r$', ...
      ' ', ...
      '$P_r = U_r(W_r^\top U_r)^{-1}W_r^\top$', ...
      ' ', ...
      '$r = f(x) - P_r f(x)$', ...
      ' ', ...
      '$W_r^\top r = 0 \;\;\Rightarrow\;\; r \perp \mathcal{W}_r$', ...
      ' ', ...
      '\color[rgb]{0.10,0.28,0.92}$\mathcal{U}_r$\color{black}: trial', ...
      '\color[rgb]{0.10,0.60,0.20}$\mathcal{W}_r$\color{black}: test'}, ...
    'Interpreter','latex', ...
    'FontSize', FS-3, ...
    'FitBoxToText','off', ...
    'BackgroundColor','white', ...
    'EdgeColor',[0.20 0.20 0.20], ...
    'LineWidth',2.0);

%% ===== Export ===========================================================
set(gcf,'Renderer','painters');
exportgraphics(gcf,'petrov_galerkin_final.pdf','ContentType','vector','BackgroundColor','white');
exportgraphics(gcf,'petrov_galerkin_final.png','Resolution',300,'BackgroundColor','white');