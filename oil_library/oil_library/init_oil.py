'''
    This is where we handle the initialization of the estimated oil properties.
    This will be the 'real' oil record that we use.

    Basically, we have an Estimated object that is a one-to-one relationship
    with the Oil object.  This is where we will place the estimated oil
    properties.
'''
import math

import transaction

from oil_library.models import ImportedRecord, Oil, KVis, Density


def process_oils(session):
    print '\nAdding Oil objects...'
    for rec in session.query(ImportedRecord):
        add_oil(rec)

    transaction.commit()


def add_oil(record):
    print 'Estimations for id {0.id}, adios_id {0.adios_oil_id}'.format(record)
    oil = Oil(name=record.oil_name)

    add_densities(record, oil)
    add_viscosities(record, oil)
    add_oil_water_interfacial_tension(record, oil)
    # TODO: should we add oil/seawater tension as well???
    add_pour_point(record, oil)
    add_flash_point(record, oil)
    record.oil = oil


def add_densities(imported_rec, oil):
    '''
        Rules:
        - If no density value exists, estimate it from the API.
          So at the end, we will always have at least one density at
          15 degrees Celsius.
        - If a density measurement at some temperature exists, but no API,
          then we estimate API from density.
          So at the end, we will always have an API value.
        - In both the previous cases, we have estimated the corollary values
          and ensured that they are consistent.  But if a record contains both
          an API and a number of densities, these values may conflict.
          In this case, we will reject the creation of the oil record.
        - TODO: This is not in the document, but Bill & Chris have verbally
          stated they would like there to always be a 15C density value.
    '''
    if len(imported_rec.densities) == 0 and imported_rec.api is not None:
        # estimate our density from api
        kg_m_3, ref_temp_k = estimate_density_from_api(imported_rec.api)

        oil.densities.append(Density(kg_m_3=kg_m_3,
                                     ref_temp_k=ref_temp_k,
                                     weathering=0.0))
        oil.api = imported_rec.api
    elif imported_rec.api is None:
        # estimate our api from density
        d_0 = density_at_temperature(imported_rec, 273.15 + 15)

        oil.api = (141.5 * 1000 / d_0) - 131.5
        for d in imported_rec.densities:
            oil.densities.append(d)
    else:
        # For now we will just accept both the api and densities
        # TODO: check if these values conflict
        oil.api = imported_rec.api

        for d in imported_rec.densities:
            oil.densities.append(d)
        pass


def estimate_density_from_api(api):
    kg_m_3 = 141.5 / (131.5 + api) * 1000
    ref_temp_k = 273.15 + 15

    return kg_m_3, ref_temp_k


def density_at_temperature(oil_rec, temperature, weathering=0.0):
    # first, get the density record closest to our temperature
    density_list = [(d, abs(d.ref_temp_k - temperature))
                    for d in oil_rec.densities
                    if d.weathering == weathering]
    if density_list:
        density_rec = sorted(density_list, key=lambda d: d[1])[0][0]
        d_ref = density_rec.kg_m_3
        t_ref = density_rec.ref_temp_k
    else:
        if oil_rec.api is None:
            # We have no densities at our requested weathering, and no api
            # We cannot make a computation.
            return None
        else:
            d_ref, t_ref = estimate_density_from_api(oil_rec.api)
    k_pt = 0.008

    # then interpolate our density based on temperature
    return d_ref / (1 - k_pt * (t_ref - temperature))


def add_viscosities(imported_rec, oil):
        '''
            Get a list of all kinematic viscosities associated with this
            oil object.  The list is compiled from the stored kinematic
            and dynamic viscosities associated with the oil record.
            The viscosity fields contain:
              - kinematic viscosity in m^2/sec
              - reference temperature in degrees kelvin
              - weathering ???
            Viscosity entries are ordered by (weathering, temperature)
            If we are using dynamic viscosities, we calculate the
            kinematic viscosity from the density that is closest
            to the respective reference temperature
        '''
        kvis = get_kvis(imported_rec)

        kvis.sort(key=lambda x: (x[2], x[1]))
        kwargs = ['m_2_s', 'ref_temp_k', 'weathering']

        for v in kvis:
            oil.kvis.append(KVis(**dict(zip(kwargs, v))))


def get_kvis(imported_rec):
    if imported_rec.kvis is not None:
        viscosities = [(k.m_2_s,
                        k.ref_temp_k,
                        (0.0 if k.weathering is None else k.weathering))
                       for k in imported_rec.kvis
                       if k.ref_temp_k is not None]
    else:
        viscosities = []

    for kv, t, w in get_kvis_from_dvis(imported_rec):
        if kvis_exists_at_temp_and_weathering(viscosities, t, w):
            continue

        viscosities.append((kv, t, w))

    return viscosities


def get_kvis_from_dvis(oil_rec):
    '''
        If we have any DVis records, we convert them to kinematic and return
        them.
        DVis records are correlated with a ref_temperature, and weathering.
        In order to convert dynamic viscosity to kinematic, we need to get
        the density at our reference temperature and weathering
    '''
    kvis_out = []

    if oil_rec.dvis:
        for dv, t, w in [(d.kg_ms,
                         d.ref_temp_k,
                         (0.0 if d.weathering is None else d.weathering))
                         for d in oil_rec.dvis]:
            density = density_at_temperature(oil_rec, t, w)

            # kvis = dvis/density
            if density is not None:
                kvis_out.append(((dv / density), t, w))

    return kvis_out


def kvis_exists_at_temp_and_weathering(kvis, temperature, weathering):
    return len([v for v in kvis
                if v[1] == temperature
                and v[2] == weathering]) > 0


def add_oil_water_interfacial_tension(imported_rec, oil):
    if imported_rec.oil_water_interfacial_tension_n_m is not None:
        oil.oil_water_interfacial_tension_n_m = \
            imported_rec.oil_water_interfacial_tension_n_m
        oil.oil_water_interfacial_tension_ref_temp_k = \
            imported_rec.oil_water_interfacial_tension_ref_temp_k
    else:
        # estimate values from api
        if imported_rec.api is not None:
            api = imported_rec.api
        elif oil.api is not None:
            api = oil.api
        else:
            api = None

        oil.oil_water_interfacial_tension_n_m = (0.001 * (39 - 0.2571 * api))
        oil.oil_water_interfacial_tension_ref_temp_k = 273.15 * 15.0
    pass


def add_pour_point(imported_rec, oil):
    '''
        If we already have pour point min-max values in our imported
        record, then we are good.  We simply copy them over.
        If we don't have them, then we will need to approximate them.

        If we have measured molecular weights for the distillation fractions
        then:
            (A) If molecular weight M_w in kg/kmol and mass fractions are
                given for all the oil fractions (j = 1...jMAX), than an
                average molecular weight for the whole oil can be estimated
                as:
                    M_w_avg = sum[1,jMAX](M_w(j) * fmass_j)
                    where fmass_j = mass fraction of component j.
                    (Note: jMAX = 2(N + 1) wherer N = number of
                           distillation cuts.  We sum over all the SARA
                           fractions but resins and asphaltenes do not
                           have distillation cut data.)
                Define SG = P_oil / 1000 kg as specific gravity.
                Define T_api = 311.15K = reference temperature for the oil
                                         kinematic viscosity.
                T_pp = (130.47 * SG^2.97) * \
                       M_w_avg^(0.61235 - 0.47357 * SG) * \
                       V_oil^(0.31 - 0.3283 * SG) * \
                       T_api
        else:
            (B) Pour point is estimated by reversing the viscosity-to-
                temperature correction in AIDOS2 and assuming that, at the
                pour point, viscosity is equal to 1 million centistokes.
    '''
    if (imported_rec.pour_point_min_k is not None or
            imported_rec.pour_point_max_k is not None):
        # we have values to copy over
        oil.pour_point_min_k = imported_rec.pour_point_min_k
        oil.pour_point_max_k = imported_rec.pour_point_max_k
    else:
        oil.pour_point_min_k = None
        if 0:
            # TODO: When would we have molecular weights?
            # if we have measured molecular weights for the
            # distillation fractions, then we use method 'A'
            # oil.pour_point_max_k = \
            #     estimate_pp_by_molecular_weights(imported_rec)
            pass
        else:
            oil.pour_point_max_k = estimate_pp_by_viscosity_ref(imported_rec)


def estimate_pp_by_viscosity_ref(imported_rec):
    # Get the viscosity measured at the lowest reference temperature
    kvis_rec = sorted(get_kvis(imported_rec),
                      key=lambda x: (x[2], x[1]))[0]

    v_ref, t_ref = kvis_rec[0], kvis_rec[1]
    c_v1 = 5000.0

    return c_v1 * t_ref / (c_v1 - t_ref * math.log(v_ref))


def add_flash_point(imported_rec, oil):
    '''
        If we already have flash point min-max values in our imported
        record, then we are good.  We simply copy them over.
        If we don't have them, then we will need to approximate them.

        If we have measured distillation cut data
        then:
            (A) T_cut1 = the boiling point of the first pseudo-component cut.
                T_flsh = 117 + 0.69 * T_cut1
        else:
            (B) T_flsh = 457 - 3.34 * api
    '''
    if (imported_rec.flash_point_min_k is not None or
            imported_rec.flash_point_max_k is not None):
        # we have values to copy over
        oil.flash_point_min_k = imported_rec.flash_point_min_k
        oil.flash_point_max_k = imported_rec.flash_point_max_k
    else:
        if 0:
            # if we have measured distillation cuts, then we use method 'A'
            oil.flash_point_min_k = estimate_fp_by_cut(imported_rec)
            oil.flash_point_max_k = estimate_fp_by_cut(imported_rec)
        else:
            # we use method 'B'
            oil.flash_point_max_k = estimate_fp_by_api(imported_rec)


def estimate_fp_by_cut(imported_rec):
    pass


def estimate_fp_by_api(imported_rec):
    pass
