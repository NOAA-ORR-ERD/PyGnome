'''
    This is where we handle the initialization of the estimated oil properties.
    This will be the 'real' oil record that we use.

    Basically, we have an Estimated object that is a one-to-one relationship
    with the Oil object.  This is where we will place the estimated oil
    properties.
'''
from math import log, log10, exp, fabs

import transaction

from oil_library.models import (ImportedRecord, Oil,
                                Density, KVis, Cut,
                                SARAFraction, MolecularWeight)
from oil_library.utilities import get_boiling_points_from_api


def process_oils(session):
    print '\nAdding Oil objects...'
    for rec in session.query(ImportedRecord):
        add_oil(rec)

    transaction.commit()


def add_oil(record):
    print 'Estimations for {0}'.format(record.adios_oil_id)
    oil = Oil()

    add_demographics(record, oil)
    add_densities(record, oil)
    add_viscosities(record, oil)
    add_oil_water_interfacial_tension(record, oil)
    # TODO: should we add oil/seawater tension as well???
    add_pour_point(record, oil)
    add_flash_point(record, oil)
    add_emulsion_water_fraction_max(record, oil)
    add_resin_fractions(oil)
    add_asphaltene_fractions(oil)
    add_bullwinkle_fractions(oil)
    add_adhesion(record, oil)
    add_sulphur_mass_fraction(record, oil)
    add_soluability(record, oil)
    add_distillation_cut_boiling_point(record, oil)
    add_molecular_weights(record, oil)
    add_component_densities(record, oil)
    add_saturate_fractions(record, oil)
    add_aromatic_fractions(record, oil)

    record.oil = oil


def add_demographics(imported_rec, oil):
    oil.name = imported_rec.oil_name


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
        - This is not in the document, but Bill & Chris have verbally
          stated they would like there to always be a 15C density value.
    '''
    for d in imported_rec.densities:
        oil.densities.append(d)

    if imported_rec.api is not None:
        oil.api = imported_rec.api
    elif oil.densities:
        # estimate our api from density
        d_0 = density_at_temperature(oil, 273.15 + 15)

        oil.api = (141.5 * 1000 / d_0) - 131.5
    else:
        print ('Warning: no densities and no api for record {0}'
               .format(imported_rec.adios_oil_id))

    if not [d for d in oil.densities if d.ref_temp_k == 273.15 + 15]:
        # add a 15C density from api
        kg_m_3, ref_temp_k = estimate_density_from_api(oil.api)

        oil.densities.append(Density(kg_m_3=kg_m_3,
                                     ref_temp_k=ref_temp_k,
                                     weathering=0.0))


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
    if density_list and fabs(t_ref - temperature) > (1 / k_pt):
        # even if we got some measured densities, they could be at
        # temperatures that is out of range for our algorithm.
        return None

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
                         for d in oil_rec.dvis
                         if d.kg_ms > 0.0]:
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

    return c_v1 * t_ref / (c_v1 - t_ref * log(v_ref))


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
        oil.flash_point_min_k = None
        if len(imported_rec.cuts) > 0:
            # if we have measured distillation cuts, then we use method 'A'
            oil.flash_point_max_k = estimate_fp_by_cut(imported_rec)
        else:
            # we use method 'B'
            oil.flash_point_max_k = estimate_fp_by_api(oil)


def estimate_fp_by_cut(imported_rec):
    '''
        If we have measured distillation cut data:
            (A) T_cut1 = the boiling point of the first pseudo-component cut.
                T_flsh = 117 + 0.69 * T_cut1
    '''
    temp_cut_1 = sorted(imported_rec.cuts,
                        key=lambda x: x.vapor_temp_k)[0].vapor_temp_k

    return 117.0 + 0.69 * temp_cut_1


def estimate_fp_by_api(imported_rec):
    '''
        If we do *not* have measured distillation cut data, then use api:
            (B) T_flsh = 457 - 3.34 * api
    '''
    return 457.0 - 3.34 * imported_rec.api


def add_emulsion_water_fraction_max(imported_rec, oil):
    '''
        This quantity will be set after the emulsification approach in ADIOS3
        is finalized.  It will vary depending upon the emulsion stability.
        For now set f_w_max = 0.9 for crude oils and f_w_max = 0.0 for
        refined products.
    '''
    if imported_rec.product_type == 'Crude':
        oil.emulsion_water_fraction_max = 0.9
    elif imported_rec.product_type == 'Refined':
        oil.emulsion_water_fraction_max = 0.0


def add_resin_fractions(oil):
    for a, b, t in get_resin_coeffs(oil):
        f_res = (0.033 * a +
                 0.00087 * b -
                 0.74)
        f_res = 0.0 if f_res < 0.0 else f_res

        oil.sara_fractions.append(SARAFraction(sara_type='Resins',
                                               fraction=f_res,
                                               ref_temp_k=t))


def add_asphaltene_fractions(oil):
    for a, b, t in get_asphaltene_coeffs(oil):
        f_asph = (0.000014 * a +
                  0.000004 * b -
                  0.18)
        f_asph = 0.0 if f_asph < 0.0 else f_asph

        oil.sara_fractions.append(SARAFraction(sara_type='Asphaltenes',
                                               fraction=f_asph,
                                               ref_temp_k=t))


def get_resin_coeffs(oil):
    '''
        Get coefficients for calculating resin (and asphaltene) fractions
        based on Merv Fingas' empirical analysis of ESTC oil properties
        database.
        Bill has clarified that we want to get the coefficients for just
        the 15C Density
        For now, we will assume we need to gather an array of coefficients
        based on the existing measured viscosities and densities.
        So we assume we are dealing with an oil object that has both.
        We return ((a0, b0),
                   (a1, b1),
                   ...
                   )
    '''
    try:
        a = [(10 * exp(0.001 * density_at_temperature(oil, k.ref_temp_k)))
             for k in oil.kvis
             if k.weathering == 0.0 and
             k.ref_temp_k == 273.15 + 15 and
             density_at_temperature(oil, k.ref_temp_k) is not None]
        b = [(10 * log(1000.0 * density_at_temperature(oil, k.ref_temp_k) *
                       k.m_2_s))
             for k in oil.kvis
             if k.weathering == 0.0 and
             k.ref_temp_k == 273.15 + 15 and
             density_at_temperature(oil, k.ref_temp_k) is not None]
    except:
        print 'generated exception for oil = ', oil
        print 'oil.kvis = ', oil.kvis
        print [(density_at_temperature(oil, k.ref_temp_k), k.m_2_s)
               for k in oil.kvis
               if k.weathering == 0.0]
        raise

    t = [k.ref_temp_k
         for k in oil.kvis
         if k.weathering == 0.0]
    return zip(a, b, t)


def get_asphaltene_coeffs(oil):
    return get_resin_coeffs(oil)


def add_bullwinkle_fractions(oil):
    '''
        This is the mass fraction that must evaporate of dissolve before
        stable emulsification can begin.
        For this estimation, we depend on an oil object with a valid
        asphaltene fraction or a valid api
        This is a scalar value calculated with a reference temperature of 15C
    '''
    f_bulls = [0.32 - 3.59 * af.fraction
               for af in oil.sara_fractions
               if af.fraction > 0 and af.sara_type == 'Asphaltenes' and
               af.ref_temp_k == 273.15 + 15]

    if not f_bulls and oil.api >= 0.0:
        f_bulls = [0.5762 * log10(oil.api)]

    if not f_bulls:
        # this can happen if we do not have an asphaltene fraction
        # (thus, viscosity or density) at 15C, and the api is 0.0 or less
        print 'Warning: could not estimate bullwinkle fractions'
        print ('\tOil(name={0.name}, sara={0.sara_fractions}, api={0.api}'
               .format(oil))
    else:
        oil.bullwinkle_fraction = f_bulls[0]


def add_adhesion(imported_rec, oil):
    '''
        This is currently not used by the model, but we will get it
        if it exists.
        Otherwise, we will assign a constant.
    '''
    if imported_rec.adhesion is not None:
        oil.adhesion_kg_m_2 = imported_rec.adhesion
    else:
        oil.adhesion_kg_m_2 = 0.035


def add_sulphur_mass_fraction(imported_rec, oil):
    '''
        This is currently not used by the model, but we will get it
        if it exists.
        Otherwise, we will assign a constant per the documentation.
    '''
    if imported_rec.sulphur is not None:
        oil.sulphur_fraction = imported_rec.sulphur
    else:
        oil.sulphur_fraction = 0.0


def add_soluability(imported_rec, oil):
    '''
        There is no direct soluability attribute in the imported record,
        so we will just assign a constant per the documentation.
    '''
    oil.sulphur_fraction = 0.0


def add_distillation_cut_boiling_point(imported_rec, oil):
    '''
        if cuts exist:
            copy them over
        else:
            get a single cut from the API
    '''
    if imported_rec.oil_name == 'ALAMO':
        print 'imported cuts:', imported_rec.cuts
    for c in imported_rec.cuts:
        oil.cuts.append(c)

    if not oil.cuts:
        # TODO: get boiling point from api inside utilities module
        mass_left = 1.0

        if imported_rec.resins:
            mass_left -= imported_rec.resins

        if imported_rec.asphaltene_content:
            mass_left -= imported_rec.asphaltene_content

        prev_mass_frac = 0.0
        for t_i, fraction in get_boiling_points_from_api(5, mass_left,
                                                         oil.api):
            oil.cuts.append(Cut(fraction=prev_mass_frac + fraction,
                                vapor_temp_k=t_i))
            prev_mass_frac += fraction

        pass


def add_saturate_fractions(imported_rec, oil):
    '''
        (A) if these hold true:
              - (i): oil library record contains summed mass fractions or
                     weight (%) for the distillation cuts
              - (ii): T(i) < 530K
            then:
              (Reference: CPPF, eq.s 3.77 and 3.78)
              - f(sat, i) = (fmass(i) *
                             (2.24 - 1.98 * SG(sat, i) - 0.009 * M(w, sat, i)))
              - if f(sat, i) >= fmass(i):
                  - f(sat, i) = fmass(i)
              - else if f(sat, i) < 0:
                  - f(sat, i) = 0
            else if these hold true:
              - (ii): T(i) >= 530K
            then:
              - f(sat, i) = fmass(i) / 2
        (B) else if there were no measured mass fractions in the imported
            record
              - apply (A) except fmass(i) = 1/5 for all cuts
    '''
    pass


def add_aromatic_fractions(imported_rec, oil):
    '''
        Reference: CPPF, eq.s 3.77 and 3.78
    '''
    pass


def add_molecular_weights(imported_rec, oil):
    for c in imported_rec.cuts:
        saturate = get_saturate_molecular_weight(c.vapor_temp_k)
        aromatic = get_aromatic_molecular_weight(c.vapor_temp_k)

        oil.molecular_weights.append(MolecularWeight(saturate=saturate,
                                                     aromatic=aromatic,
                                                     ref_temp_k=c.vapor_temp_k)
                                     )


def get_saturate_molecular_weight(vapor_temp):
    '''
        (Reference: CPPF, eq. 2.48 and table 2.6)
    '''
    if vapor_temp < 1070.0:
        return (49.7 * (6.983 - log(1070.0 - vapor_temp))) ** (3. / 2.)
    else:
        return None


def get_aromatic_molecular_weight(vapor_temp):
    '''
        (Reference: CPPF, eq. 2.48 and table 2.6)
    '''
    if vapor_temp < 1015.0:
        return (44.5 * (6.91 - log(1015.0 - vapor_temp))) ** (3. / 2.)
    else:
        return None


def add_component_densities(imported_rec, oil):
    '''
        (Reference: CPPF, eq. 2.13 and table 9.6)
    '''
    P_asph = P_res = 1100.0  # kg/m^3

    # Watson characterization factors
    K_arom = 10.0
    K_sat = 12.0
    pass
