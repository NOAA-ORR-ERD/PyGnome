#ifndef __PYX_HAVE__gnome__c_gnome
#define __PYX_HAVE__gnome__c_gnome


/* "C:\Users\brian.zelenke\Documents\GNOME\GIT\GNOME\py_gnome\cyGNOME\c_gnome_defs.pxi":166
 *         OSErr            ComputeVelocityScale()
 * 
 * cdef public enum type_defs:             # <<<<<<<<<<<<<<
 *     status_not_released = OILSTAT_NOTRELEASED
 *     status_in_water = OILSTAT_INWATER
 */
enum type_defs {

  /* "C:\Users\brian.zelenke\Documents\GNOME\GIT\GNOME\py_gnome\cyGNOME\c_gnome_defs.pxi":180
 *     disp_status_have_evaporated = HAVE_EVAPORATED
 *     disp_status_remove = REMOVE
 *     disp_status_have_removed = HAVE_REMOVED             # <<<<<<<<<<<<<<
 */
  status_not_released = OILSTAT_NOTRELEASED,
  status_in_water = OILSTAT_INWATER,
  status_on_land = OILSTAT_ONLAND,
  status_off_maps = OILSTAT_OFFMAPS,
  status_evaporated = OILSTAT_EVAPORATED,
  disp_status_dont_disperse = DONT_DISPERSE,
  disp_status_disperse = DISPERSE,
  disp_status_have_dispersed = HAVE_DISPERSED,
  disp_status_disperse_nat = DISPERSE_NAT,
  disp_status_have_dispersed_nat = HAVE_DISPERSED_NAT,
  disp_status_evaporate = EVAPORATE,
  disp_status_have_evaporated = HAVE_EVAPORATED,
  disp_status_remove = REMOVE,
  disp_status_have_removed = HAVE_REMOVED
};

#ifndef __PYX_HAVE_API__gnome__c_gnome

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#endif /* !__PYX_HAVE_API__gnome__c_gnome */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initc_gnome(void);
#else
PyMODINIT_FUNC PyInit_c_gnome(void);
#endif

#endif /* !__PYX_HAVE__gnome__c_gnome */
