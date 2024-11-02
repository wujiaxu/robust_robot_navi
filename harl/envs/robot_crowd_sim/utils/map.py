from typing import List
from shapely.geometry import MultiPolygon, LinearRing,Point

class Map:
    obstacles:List
    def __init__(self,width,hight):
        self._map_width = width
        self._map_hight = hight
        #consider the map shape is square in default
        self._map_boundary = LinearRing( ((-self._map_width/2., self._map_hight/2.), 
                                    (self._map_width/2., self._map_hight/2.),
                                    (self._map_width/2., -self._map_hight/2.),
                                    (-self._map_width/2.,-self._map_hight/2.)) )
        
    def getBoundary(self):

        return self._map_boundary.coords.xy
    
    def checkCollision(self,agent):
        return self._map_boundary.intersects(agent.collider)
    
    def checkDistance(self,agent):
        return self._map_boundary.distance(agent.collider)