%% galerkin_final.m
%  KEY FIX: plane spans only trajectory footprint (tight).
%            View limits are wider to show vectors/labels.
% -------------------------------------------------------------------------
clear; close all; clc;

%% ===== Style ============================================================
LW_curve3D = 3.2;
LW_curve2D = 3.0;
LW_dash    = 1.6;
LW_axis    = 1.8;
LW_vec     = 3.8;
LW_box     = 1.6;

axisBlue  = [0.08 0.20 0.88];
resRed    = [0.85 0.06 0.06];
projBlue  = [0.08 0.28 0.92];
planeCol  = [0.91 0.91 0.91];

FS = 19;

%% ===== Smooth 3D trajectory =============================================
t  = linspace(0, 2*pi, 500);

a1 = 0.50 + 0.55*cos(t)     + 0.14*cos(2.1*t);
a2 = 0.40 + 0.80*sin(0.9*t) + 0.10*sin(3.0*t);
a3 = 1.20 + 0.38*sin(1.4*t) + 0.10*cos(2.5*t);

a1p = a1;
a2p = a2;
a3p = zeros(size(a1));

%% ===== Evaluation point =================================================
[~, k0] = max(a3(50:350));
k0 = k0 + 49;

x_full = [a1(k0); a2(k0); a3(k0)];

% f(x): shallow diagonal (~30 deg off plane)
VSCALE  = 1.5;
fx_dir  = [0.88; -0.18; 0.50];
fx_dir  = fx_dir / norm(fx_dir);
fx      = VSCALE * fx_dir;

fx_proj = [fx(1); fx(2); 0];
r       = fx - fx_proj;

tip_f  = x_full + fx;
tip_Pf = x_full + fx_proj;
mid_r  = tip_Pf + 0.50*r;

%% ===== SEPARATE limits: plane (tight) vs view (roomy) ===================

% ── Plane spans ONLY the trajectory footprint ──────────────────────────
planePad  = 0.18;
xL_plane  = [min(a1) - planePad,  max(a1) + planePad];
yL_plane  = [min(a2) - planePad,  max(a2) + planePad];

% ── View limits give room for vectors, labels, axis arrows ──────────────
xpad = 0.65;   ypad = 0.65;   zpad = 0.75;
xL   = [min(a1) - xpad,  max([a1(:)', tip_f(1), tip_Pf(1)]) + xpad];
yL   = [min(a2) - ypad,  max([a2(:)', tip_f(2), tip_Pf(2)]) + ypad];

a3_axlen = max(a3(:)) + 0.25;
zL       = [-0.10,  max([a3(:)', tip_f(3)]) + zpad];

%% ===== Figure ============================================================
hFig = figure('Color','w','Units','inches','Position',[0.5 0.5 15 8.5]);
hold on;

ax = gca;
ax.Units = 'normalized';
ax.Position = [0.02  0.04  0.73  0.93];
ax.ActivePositionProperty = 'position';

%% ===== Horizontal plane — TIGHT size ====================================
[xp,yp] = meshgrid(xL_plane, yL_plane);   % <-- uses tight limits
surf(xp, yp, zeros(2,2), ...
     'FaceColor',planeCol,'FaceAlpha',0.55, ...
     'EdgeColor',[0.50 0.50 0.50],'LineWidth',1.0);

%% ===== Coordinate axes — anchored to PLANE corners ======================
O      = [xL_plane(1), yL_plane(1), 0];
ax2end = [xL_plane(2)+0.10,  yL_plane(1),       0        ];
ax1end = [xL_plane(1),       yL_plane(2)+0.10,  0        ];
ax3end = [xL_plane(1),       yL_plane(1),        a3_axlen ];

plot3([O(1) ax2end(1)],[O(2) ax2end(2)],[0 0],            'k','LineWidth',LW_axis);
plot3([O(1) ax1end(1)],[O(2) ax1end(2)],[0 0],            'k','LineWidth',LW_axis);
plot3([O(1) ax3end(1)],[O(2) ax3end(2)],[O(3) a3_axlen],  'k','LineWidth',LW_axis);

quiver3(ax2end(1),ax2end(2),0,        0.001,0,    0,    0,'k','LineWidth',LW_axis,'MaxHeadSize',50);
quiver3(ax1end(1),ax1end(2),0,        0,    0.001,0,    0,'k','LineWidth',LW_axis,'MaxHeadSize',50);
quiver3(ax3end(1),ax3end(2),a3_axlen, 0,    0,    0.001,0,'k','LineWidth',LW_axis,'MaxHeadSize',50);

text(ax2end(1)+0.07, ax2end(2),       0,             '$a_2$','FontSize',FS+1,'Interpreter','latex','Color',axisBlue);
text(ax1end(1),      ax1end(2)+0.07,  0,             '$a_1$','FontSize',FS+1,'Interpreter','latex','Color',axisBlue);
text(ax3end(1)-0.20, ax3end(2),       a3_axlen+0.04, '$a_3$','FontSize',FS+1,'Interpreter','latex','Color',axisBlue);

%% ===== 2D projected trajectory (blue) ===================================
plot3(a1p, a2p, a3p, 'Color',axisBlue,'LineWidth',LW_curve2D);

for ki = round(linspace(80, 420, 3))
    d2 = [a1p(ki+10)-a1p(ki-10), a2p(ki+10)-a2p(ki-10), 0];
    d2 = d2/norm(d2)*0.001;
    quiver3(a1p(ki),a2p(ki),0, d2(1),d2(2),0, 0, ...
            'Color',axisBlue,'LineWidth',2.6,'MaxHeadSize',50);
end

%% ===== Vertical drop lines ==============================================
dropIdx = round(linspace(40, 460, 10));
for k = dropIdx
    plot3([a1(k) a1(k)],[a2(k) a2(k)],[0 a3(k)],'--', ...
          'Color',[0.38 0.38 0.38],'LineWidth',LW_dash);
end

%% ===== 3D trajectory (black) ============================================
plot3(a1, a2, a3, 'k','LineWidth',LW_curve3D);
plot3(a1(1),a2(1),a3(1),'ko','MarkerFaceColor','k','MarkerSize',8);

for ki = round(linspace(70, 430, 3))
    d3 = [a1(ki+10)-a1(ki-10), a2(ki+10)-a2(ki-10), a3(ki+10)-a3(ki-10)];
    d3 = d3/norm(d3)*0.001;
    quiver3(a1(ki),a2(ki),a3(ki), d3(1),d3(2),d3(3), 0, ...
            'Color','k','LineWidth',3.0,'MaxHeadSize',50);
end

%% ===== Evaluation point =================================================
plot3(x_full(1),x_full(2),x_full(3),'ko','MarkerFaceColor','k','MarkerSize',10);
text(x_full(1)-0.38, x_full(2)-0.05, x_full(3)+0.09, ...
     '$x(t)$','FontSize',FS,'Interpreter','latex','Color','k');

%% ===== VECTORS ==========================================================

% f(x) — black
quiver3(x_full(1),x_full(2),x_full(3), fx(1),fx(2),fx(3), 0, ...
        'Color','k','LineWidth',LW_vec,'MaxHeadSize',0.55);
text(tip_f(1)+0.06, tip_f(2)+0.02, tip_f(3)+0.07, ...
     '$f(x)$','FontSize',FS+1,'Interpreter','latex', ...
     'Color','k','FontWeight','bold');

% P_r f(x) — blue
quiver3(x_full(1),x_full(2),x_full(3), fx_proj(1),fx_proj(2),fx_proj(3), 0, ...
        'Color',projBlue,'LineWidth',LW_vec,'MaxHeadSize',0.55);
text(tip_Pf(1)+0.06, tip_Pf(2)-0.07, tip_Pf(3)-0.20, ...
     '$P_r f(x)$','FontSize',FS+1,'Interpreter','latex', ...
     'Color',projBlue,'FontWeight','bold');

% Residual r — red
r_draw = tip_f - tip_Pf;
quiver3(tip_Pf(1),tip_Pf(2),tip_Pf(3), r_draw(1),r_draw(2),r_draw(3), 0, ...
        'Color',resRed,'LineWidth',LW_vec,'MaxHeadSize',0.55);
text(mid_r(1)+0.14, mid_r(2)+0.02, mid_r(3), ...
     '$r$','FontSize',FS+4,'Interpreter','latex', ...
     'Color',resRed,'FontWeight','bold');

% Dashed connector
plot3([tip_f(1) tip_Pf(1)],[tip_f(2) tip_Pf(2)],[tip_f(3) tip_Pf(3)], ...
      '--','Color',[0.45 0.55 0.95],'LineWidth',LW_dash+0.4);

% Right-angle box
d_Pf = fx_proj / norm(fx_proj);
d_r  = r / norm(r);
sq   = 0.16;
bx   = [tip_Pf, tip_Pf+sq*d_Pf, tip_Pf+sq*(d_Pf+d_r), tip_Pf+sq*d_r, tip_Pf];
plot3(bx(1,:),bx(2,:),bx(3,:),'k-','LineWidth',LW_box);

%% ===== View =============================================================
xlim(xL); ylim(yL); zlim(zL);
axis off; grid off;
axis vis3d;
daspect([1 1 0.75]);
view(132, 26);
camproj('perspective');
camzoom(0.88);    % slightly zoomed out — all elements visible

%% ===== Equation box =====================================================
annotation('textbox',[0.76 0.38 0.22 0.38], ...
    'String',{ ...
      '\bf Galerkin condition:', ...
      ' ', ...
      '$P_r = U_r U_r^\top$', ...
      ' ', ...
      '$r = f(x) - P_r f(x)$', ...
      ' ', ...
      '$U_r^\top\, r = 0$', ...
      ' ', ...
      '$r \perp \mathcal{U}_r$'}, ...
    'Interpreter','latex', ...
    'FontSize', FS-2, ...
    'FitBoxToText','off', ...
    'BackgroundColor','white', ...
    'EdgeColor',[0.20 0.20 0.20], ...
    'LineWidth',2.0);

%% ===== Export ===========================================================
set(gcf,'Renderer','painters');
hFig.PaperUnits    = 'inches';
hFig.PaperSize     = [15 8.5];
hFig.PaperPosition = [0 0 15 8.5];

exportgraphics(gcf,'galerkin_final.pdf', ...
    'ContentType','vector','BackgroundColor','white');
exportgraphics(gcf,'galerkin_final.png', ...
    'Resolution',300,'BackgroundColor','white');