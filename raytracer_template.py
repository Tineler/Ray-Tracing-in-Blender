import bpy
import mathutils
import math

##############################################################
# ADDON INFO
##############################################################
bl_info = {
            "name": "Raytracer",
            "author": "Matthias Buechi, Martin Haas",
            "category": "Render"
            }


##############################################################
#FUNCTION DEFINITIONS
##############################################################
def multiply_vector_components(a, b):
    """ Componentwise multiplication of two Vectors.
        a = (1, 2, 3), b = (2, 2, 2), a -> (2, 4, 6)
    """
    for i in range(len(a)):
        a[i] *= b[i]

def cross_2d(vec_a, vec_b, index_i, index_j):
    """ Calculates kind of 2D crossproduct """

    return vec_a[index_i] * vec_b[index_j] - vec_a[index_j] * vec_b[index_i]


def interpolate_normal(vertices, normals, point, polygon_normal):
    if len(vertices) == 4:
        return interpolate_normal_quad(vertices, normals, point, polygon_normal)
    elif len(vertices) == 3:
        return interpolate_normal_triangle(vertices, normals, point)
    else:
        return None


def interpolate_normal_quad(vertices, normals, point, polygon_normal):
    """ interpolation of a points normal within a quad in 3D.
        transforms the quad from 3D to 2D and uses the function 'interpolate_normal_quad_2d' to interpolate the normal

        vertices        -- four vertices which form a quad
        normals         -- the normals of the vertices
        point           -- the point the normal is searched for
        polygon_normal  -- the surface normal of the quad
    """

    if len(vertices) != 4:
        return None

    tangent = vertices[1] - vertices[0]
    bitangent = tangent.cross(polygon_normal)

    tangent.normalize()
    bitangent.normalize()
    normal = polygon_normal.normalized()

    tbn_matrix = mathutils.Matrix().to_3x3()
    tbn_matrix[0] = tangent
    tbn_matrix[1] = bitangent
    tbn_matrix[2] = normal

    rotated_vertices = []
    rotated_point = tbn_matrix * point

    for vertex in vertices:
        rotated_vertices.append(tbn_matrix * vertex)

    return interpolate_normal_quad_2d(rotated_vertices, normals, rotated_point)


def interpolate_normal_quad_2d(vertices, normals, point):
    """ bilinear interpolation of a points normal within a quad in 2D.
        (see http://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm)

        vertices -- four vertices which form a quad
        normals  -- the normals of the vertices
        point    -- the point the normal is searched for
    """

    if len(vertices) != 4:
        return None

    e = vertices[1] - vertices[0]
    f = vertices[3] - vertices[0]
    g = vertices[0]- vertices[1] + vertices[2] - vertices[3]
    h = point - vertices[0]

    i = 0
    j = 1

    k2 = cross_2d(g, f, i, j)
    k1 = cross_2d(e, f, i, j) + cross_2d(h, g, i, j)
    k0 = cross_2d(h, e, i, j)

    if k2 == 0:
        # The quad is a perfect rectangle
        v = -(k0/k1)
    else:
        down = (k1 ** 2)-(4.0*k0*k2)
        if down < 0:
            # The Point 'point' isn't within the quad
            print("Point is not in the quad")
            return None
        root = math.sqrt(down)

        v1 = (-k1 - root) / (2.0*k2)

        if v1 < 0 or v1 > 1:
            v = (-k1 + root) / (2.0*k2)
        else:
            v = v1

    u = (h[i] - f[i]*v)/(e[i]+g[i]*v)

    p_normal = (1.0-u) * normals[0] + u * normals[1]
    q_normal = (1.0-u) * normals[3] + u * normals[2]

    normal = (1.0-v) * p_normal + v * q_normal
    normal.normalize()

    return normal


def interpolate_normal_triangle(vertices, normals, point):
    """ interpolates the normal for a point within a triangle.
        uses barycentric coordinates

        vertices -- three vertices which form a quad
        normals  -- the normals of the vertices
        point    -- the point the normal is searched for
    """

    if len(vertices) != 3:
        return None

    ab = vertices[1] - vertices[0]
    ac = vertices[2] - vertices[0]
    bc = vertices[2] - vertices[1]
    aq = point - vertices[0]
    bq = point - vertices[1]

    abq = (ab.cross(aq)).length / 2
    acq = (ac.cross(aq)).length / 2
    bcq = (bc.cross(bq)).length / 2

    abc = (ab.cross(ac)).length / 2

    ratio_a = bcq / abc
    ratio_b = acq / abc
    ratio_c = abq / abc

    normal = normals[0] * ratio_a + normals[1] * ratio_b + normals[2] * ratio_c
    return normal


##############################################################
#CLASS DEFINITIONS
##############################################################
class RaytracerRenderEngine(bpy.types.RenderEngine):
    bl_idname = 'raytracer'
    bl_label = 'Raytrace Renderer'
    bl_use_preview = False

    def render(self, scene):
        if scene.name != 'preview':
            self.init_rendering(scene)
            self.init_sampling(scene)
            self.start_rendering(scene)

    def init_rendering(self, scene):
        scale = scene.render.resolution_percentage / 100.0

        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        camera = scene.camera
        projection_plane = camera.data.view_frame(scene)

        self.projection_plane_w = math.fabs(projection_plane[0][0]) * 2
        self.projection_plane_h = math.fabs(projection_plane[0][1]) * 2
        self.distance_to_projection_plane = math.fabs(projection_plane[0][2])

        self.center_of_projection = camera.location

        self.pixel_width = self.projection_plane_w / self.size_x
        self.pixel_height = self.projection_plane_h / self.size_y

    def init_sampling(self, scene):
        """ initialize sampling for anti aliasing """

        self.nr_of_samples = 1

        if scene.render.use_antialiasing:
            self.nr_of_samples = float(scene.render.antialiasing_samples)

        self.sample_sub_div = int(math.sqrt(self.nr_of_samples)) + 1

        self.sample_identation_y = self.pixel_height / self.sample_sub_div
        self.sample_identation_x = self.pixel_width / self.sample_sub_div

        self.sample_color_multiplier =  1.0 / self.nr_of_samples

    def start_rendering(self, scene):
        raytracer = Raytracer(scene)
        x_range = (int( self.size_x * scene.render.border_min_x), int( self.size_x * scene.render.border_max_x))
        y_range = (int( self.size_y * scene.render.border_min_y), int( self.size_y * scene.render.border_max_y))
        calc_y = y_range[1] - y_range[0]

        for y in range(y_range[0], y_range[1]):
            pixel_buffer = []
            y_view = (self.projection_plane_h/2) - ((y_range[1]-y)*self.pixel_height + self.pixel_height/2)

            for x in range(x_range[0], x_range[1]):
                x_view = -(self.projection_plane_w/2) + (x*self.pixel_width + self.pixel_width/2)
                dCamera = mathutils.Vector((x_view, y_view, -self.distance_to_projection_plane))

                dCamera.rotate(scene.camera.matrix_world)

                ray = Ray(self.center_of_projection, dCamera)
                pixel = raytracer.trace(ray)
                pixel_buffer.append(mathutils.Vector(pixel).to_4d())

                # check for render abortion
                if self.test_break():
                    return

            self.set_pixel_row(y + y_range[0], pixel_buffer, calc_y)

    def set_pixel_row(self, row_nr, pixels, calc_y):
        result = self.begin_result(0, row_nr, len(pixels), 1)
        layer = result.layers[0]
        layer.rect = pixels
        self.end_result(result)
        self.update_progress((1.0 / calc_y) * row_nr)

class Raytracer(object):

    NOT_RENDERABLE_OBJECTS = ['LAMP', 'CAMERA', 'EMPTY', 'META', 'ARMATURE', 'LATTICE']

    def __init__(self, scene, max_recursion_level = 3):
        self.scene = scene
        self.max_recursion_level = max_recursion_level
        self.objects = []
        self.lights = []
        self.test_outputs = 10

        for object in scene.objects:
            if not object.type in self.NOT_RENDERABLE_OBJECTS and not object.hide_render:
                self.objects.append(object)
            elif object.type in ['LAMP']:
                self.lights.append(object)

    def trace(self, ray):
        intersector = None
        for object in self.objects:
            newIntersect = self.get_intersection(object, ray)
            if newIntersect is not None:
                if intersector is None:
                    intersector = self.get_intersection(object, ray)
                else:
                    oldDistance = self.scene.camera.location - intersector.get_location()
                    newDistance = self.scene.camera.location - newIntersect.get_location()
                    if newDistance < oldDistance:
                        intersector = self.get_intersection(object, ray)

        if intersector is not None:
            diffuse = mathutils.Vector(intersector.object.active_material.diffuse_color)
            ambient = mathutils.Vector(self.scene.world.ambient_color)

            color = mathutils.Color((diffuse.x*ambient.x, diffuse.y*ambient.y, diffuse.z*ambient.z))

            for light in self.lights:
                lightDirection = (light.location - intersector.get_location()).normalized()
                normal = intersector.get_normal().normalized()

                if(normal * lightDirection >= 0):
                    #Calculate diffused light
                    diffuseFactor = normal * lightDirection
                    diffuseWithFactor = diffuse*diffuseFactor
                    resultDiffuse = mathutils.Color((diffuseWithFactor.x*light.data.color.r, diffuseWithFactor.y*light.data.color.g, diffuseWithFactor.z*light.data.color.b))

                    color = color + resultDiffuse

                    #Calculate specular reflection
                    specular = mathutils.Vector(intersector.object.active_material.specular_color)
                    view = (self.scene.camera.location - intersector.get_location()).normalized()

                    reflection = 2 * normal * (normal.dot(lightDirection)) - lightDirection
                    specularFactor = pow((reflection.dot(view)), intersector.object.active_material.specular_hardness)

                    specularWithFactor =specular*specularFactor
                    resultSpecular = mathutils.Color((specularWithFactor.x*light.data.color.r, specularWithFactor.y*light.data.color.g, specularWithFactor.z*light.data.color.b))

                    color = color + resultSpecular
        else:
            color = self.scene.world.horizon_color.copy()

        return color

    def is_object_between(self, object, start, end):
        """ returns True if 'object' is between points start and end """

        direction = end - start
        # move start, otherwise probably an intersection is found at the start point with the same object
        moved_start = start + 0.01 * direction
        local_start = object.matrix_world.inverted() * moved_start
        local_end = object.matrix_world.inverted() * end

        ret = object.ray_cast(local_start, local_end)

        return (ret[2] != -1)

    def get_intersection(self, object, ray):
        """ Returns an Intersection Instance, if a intersection is found """

        (line_start, line_end) = ray.get_line()
        local_start = object.matrix_world.inverted() * line_start
        local_end = object.matrix_world.inverted() * line_end

        (location, normal, index) = object.ray_cast(local_start, local_end)

        if index != -1:
            return Intersection(object, location, normal, index)

        return None

class Ray(object):

    def __init__(self, origin, direction, level = 0, ior_history = [(-1, 1.000292)]):
        self.origin = origin
        self.direction = direction.normalized()
        self.level = level
        self.ior_history = ior_history

    def get_line(self):
        """ returns a line, which can be used for the ray_cast method of an object
        """
        start = self.origin + (0.0001 * self.direction)
        end = self.origin + (1000 * self.direction)
        return (start, end)

    def info(self):
        print("----------------------------------------")
        print("start: x = %.10f, y = %.10f, z = %.10f " % (self.origin[0], self.origin[1], self.origin[2]))
        print("direction: x = %.10f, y = %.10f, z = %.10f " % (self.direction[0], self.direction[1], self.direction[2]))

    def add_ior(self, object, ior):
        self.ior_history.append((object.data.name, ior))

    def remove_ior(self, object):
        toRemove = -1
        for index, (obj_name, ior) in enumerate(self.ior_history):
            toRemove = index

        if toRemove > 0:
            del self.ior_history[index]

    def get_current_ior(self):
        return self.ior_history[len(self.ior_history)-1][1]

class Intersection(object):

    def __init__(self, object, location, normal, face_index):
        self.object = object
        self.location = location
        self.normal = normal
        self.face_index = face_index

    def get_normal(self, world_space = True):
        polygon = self.object.data.polygons[self.face_index]
        normal = self.normal.copy()

        #interpolate normal if smooth is activated
        if len(self.object.data.polygons) > 1 and polygon.use_smooth:
            normals = []
            vertices = []

            for index in polygon.vertices:
                vertex = self.object.data.vertices[index]
                vertices.append(mathutils.Vector(vertex.co))
                normals.append(vertex.normal)

            interpolated_normal = interpolate_normal(vertices, normals, self.location, polygon.normal)

            if interpolated_normal:
                normal = interpolated_normal

        if world_space:
            normal.rotate(self.object.matrix_world)

        return normal.normalized()

    def get_location(self, world_space = True):
        if not world_space:
            return self.location.copy()
        else:
            return self.object.matrix_world * self.location.copy()

##############################################################
#DEFINE PROPERTIES, SHOWN IN THE UI FOR THIS RENDER ENGINE
##############################################################
from bl_ui import properties_render
properties_render.RENDER_PT_render.COMPAT_ENGINES.add('raytracer')
properties_render.RENDER_PT_dimensions.COMPAT_ENGINES.add('raytracer')
properties_render.RENDER_PT_antialiasing.COMPAT_ENGINES.add('raytracer')
del properties_render

from bl_ui import properties_material
properties_material.MATERIAL_PT_context_material.COMPAT_ENGINES.add('raytracer')
properties_material.MATERIAL_PT_diffuse.COMPAT_ENGINES.add('raytracer')
properties_material.MATERIAL_PT_specular.COMPAT_ENGINES.add('raytracer')
properties_material.MATERIAL_PT_transp.COMPAT_ENGINES.add('raytracer')
properties_material.MATERIAL_PT_mirror.COMPAT_ENGINES.add('raytracer')
del properties_material

from bl_ui import properties_world
properties_world.WORLD_PT_world.COMPAT_ENGINES.add('raytracer')
del properties_world

from bl_ui import properties_data_lamp
properties_data_lamp.DATA_PT_lamp.COMPAT_ENGINES.add('raytracer')
del properties_data_lamp

##############################################################
#REGISTER/UNREGISTER RENDERENGINE
##############################################################

def register():
    bpy.utils.register_class(RaytracerRenderEngine)

def unregister():
    bpy.utils.unregister_class(RaytracerRenderEngine)

if __name__ == "__main__":
    register()
