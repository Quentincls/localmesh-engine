"""Les quatre caméras : d'abord celles de l'exemple Pixal MV, puis les vraies.

L'exemple publié pose quatre caméras à 0, 90, 180 et 270 degrés, même champ,
même distance. Tant que les photos viennent d'un tour régulier, c'est juste.

CE QUI SE PASSE QUAND ELLES N'EN VIENNENT PAS. Mesure du 9 septembre 2026 sur
les quatre photos du samouraï de Quentin, prises à la main autour de la
figurine, les deux têtes de Depth Anything 3 d'accord à un demi-degré près :

    face      0,0 deg      attendu    0     écart      0
    droite   97,7 deg      attendu   90     écart    7,7
    dos     173,4 deg      attendu  180     écart    6,6
    gauche  -53,6 deg      attendu  -90     écart   36,4   <--

Entre vues successives : 97, 76, 133 et 54 degrés. Pas un tour régulier.

Projeter la vue de gauche comme si elle était à -90 degrés fait chercher, pour
chaque cellule de la grille, le mauvais pixel de cette photo — décalé de 36
degrés de rotation. La fusion reçoit alors deux objets superposés là où cette
vue en recouvre une autre, et sculpte les deux : sur ce sujet, une cavité
béante dans le dos, sous la cape, pleine d'éclats.

D'où `azimut` : quand la mesure est fiable, on projette depuis l'angle où la
photo a VRAIMENT été prise. Sinon on retombe sur l'angle rond, et le résultat
le dit.

CE QUI RESTE NOMINAL, ET C'EST UNE LIMITE ASSUMEE : le champ, la distance et
l'ELEVATION. Les quatre élévations mesurées ici tiennent dans deux degrés ;
un tour photographié en plongée sortirait toujours faux, et ce module ne sait
pas encore le corriger.
"""
import math
import torch

FOV = math.radians(20.)
DISTANCE = 3.1192049980163574
AZIMUTHS = {'front':0., 'right':90., 'back':180., 'left':270.}

def nominal_queries(coords, role, grid_resolution=32, image_resolution=1024,
                    inverse_scale=1., inverse_center=None, image_transform=None,
                    azimut=None):
    """Où chaque cellule de la grille tombe dans cette vue.

    `azimut`, en degrés, remplace l'angle rond du rôle quand on a mesuré d'où
    la photo vient. Voir l'en-tête du module pour la mesure qui l'impose.
    """
    if azimut is None and role not in AZIMUTHS:
        raise ValueError('Unknown view role')
    axis = torch.linspace(-1,1,grid_resolution,device=coords.device)/2
    p = axis[coords[:,1:].long()]
    p = p * inverse_scale
    if inverse_center is not None:
        p = p + torch.as_tensor(inverse_center, device=p.device, dtype=p.dtype)
    yaw = math.radians(AZIMUTHS[role] if azimut is None else azimut)
    x_camera = math.cos(yaw)*p[:,0]-math.sin(yaw)*p[:,2]
    y_camera = p[:,1]
    depth = DISTANCE-math.sin(yaw)*p[:,0]-math.cos(yaw)*p[:,2]
    focal = image_resolution/(2*math.tan(FOV/2))
    x = focal*x_camera/(depth+1e-8)+image_resolution/2
    y = -focal*y_camera/(depth+1e-8)+image_resolution/2
    grid=(torch.stack((x,y),-1)+.5)/image_resolution*2-1
    if image_transform is not None:
        # Only when the declared nominal camera describes the original image.
        grid=grid*torch.as_tensor(image_transform['normalized_scale'],device=grid.device,dtype=grid.dtype)+torch.as_tensor(image_transform['normalized_offset'],device=grid.device,dtype=grid.dtype)
    return grid
