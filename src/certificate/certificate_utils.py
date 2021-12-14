from src.certificate.barrier_certificate import BarrierCertificate
from src.certificate.barrier_certificate_alternative import BarrierCertificateAlternative
from src.certificate.lyapunov_certificate import LyapunovCertificate
from src.shared.consts import CertificateType

def get_certificate(certificate: CertificateType):
    if certificate == CertificateType.LYAPUNOV:
        return LyapunovCertificate
    if certificate == CertificateType.BARRIER:
        return BarrierCertificate
    if certificate == CertificateType.BARRIER_ALTERNATE:
        return BarrierCertificateAlternative