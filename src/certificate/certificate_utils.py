from src.certificate.barrier_certificate import BarrierCertificate
from src.certificate.barrier_certificate_alternative import BarrierLyapunovCertificate
from src.certificate.lyapunov_certificate import LyapunovCertificate
from src.certificate.rws_certificate import ReachWhileStayCertificate
from src.shared.consts import CertificateType

def get_certificate(certificate: CertificateType):
    if certificate == CertificateType.LYAPUNOV:
        return LyapunovCertificate
    if certificate == CertificateType.BARRIER:
        return BarrierCertificate
    if certificate == CertificateType.BARRIER_LYAPUNOV:
        return BarrierLyapunovCertificate
    if certificate == CertificateType.RWS:
        return ReachWhileStayCertificate